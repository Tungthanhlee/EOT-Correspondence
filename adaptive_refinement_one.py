# running on a single shape pair
import os
import scipy.io as sio
from tqdm.auto import tqdm
import shutil
import time
import argparse
from typing import Tuple, Dict, Any, Optional

import torch
import torch.optim as optim
import torch.nn.functional as F

from networks.diffusion_network import DiffusionNet
from networks.permutation_network import Similarity
from networks.fmap_network import RegularizedFMNet

from utils.shape_util import read_shape
from utils.texture_util import write_obj_pair
from utils.geometry_util import compute_operators
from utils.fmap_util import nn_query, fmap2pointmap
from utils.tensor_util import to_numpy

from losses.fmap_loss import SURFMNetLoss, PartialFmapsLoss, SquaredFrobeniusLoss
from losses.dirichlet_loss import DirichletLoss
from losses.ot_loss import ISEBSW, SW


class AdaptiveRefinement:
    """adaptive refinement of shape correspondence."""
    
    def __init__(self, args: argparse.Namespace):
        """Initialize the adaptive refinement process.
        
        Args:
            args: Command line arguments
        """
        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load shapes
        self._load_shapes()
        
        # Compute operators
        self._compute_spectral_operators()
        
        # Initialize networks
        self._init_networks()
        
        # Initialize losses
        self._init_losses()
        
    def _load_shapes(self) -> None:
        """Load source and target shapes."""
        # Read shapes
        self.vert_np_x, self.face_np_x = read_shape(self.args.path_src_shape)
        self.vert_np_y, self.face_np_y = read_shape(self.args.path_tar_shape)
        
        # Convert to tensors
        self.vert_x, self.face_x = self._to_tensor(self.vert_np_x, self.face_np_x)
        self.vert_y, self.face_y = self._to_tensor(self.vert_np_y, self.face_np_y)
    
    def _to_tensor(self, vert_np: Any, face_np: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert numpy arrays to tensors.
        
        Args:
            vert_np: Vertex coordinates as numpy array
            face_np: Face indices as numpy array
            
        Returns:
            Tuple of vertex and face tensors
        """
        vert = torch.from_numpy(vert_np).to(device=self.device, dtype=torch.float32)
        face = torch.from_numpy(face_np).to(device=self.device, dtype=torch.long)
        return vert, face
    
    def _compute_spectral_operators(self) -> None:
        """Compute Laplacian and spectral operators for both shapes."""
        _, self.mass_x, self.Lx, self.evals_x, self.evecs_x, _, _ = compute_operators(
            self.vert_x, self.face_x, k=200
        )
        _, self.mass_y, self.Ly, self.evals_y, self.evecs_y, _, _ = compute_operators(
            self.vert_y, self.face_y, k=200
        )
        
        self.evecs_trans_x = self.evecs_x.T * self.mass_x[None]
        self.evecs_trans_y = self.evecs_y.T * self.mass_y[None]
    
    def _init_networks(self) -> None:
        """Initialize neural networks."""
        # Feature extractor
        input_type = 'wks'
        in_channels = 128 if input_type == 'wks' else 3
        self.feature_extractor = DiffusionNet(
            in_channels=in_channels, 
            out_channels=256, 
            input_type=input_type
        ).to(self.device)
        
        # Load pretrained weights
        self.feature_extractor.load_state_dict(
            torch.load(self.args.checkpoint)['networks']['feature_extractor'], 
            strict=True
        )
        self.feature_extractor.eval()
        
        # Permutation network
        self.permutation = Similarity(tau=0.07, hard=True).to(self.device)
        
        # Functional map network (for refinement)
        if self.args.num_refine > 0:
            self.fmap_net = RegularizedFMNet(bidirectional=True)
            self.optimizer = optim.Adam(self.feature_extractor.parameters(), lr=1e-3)
    
    def _init_losses(self) -> None:
        """Initialize loss functions."""
        if self.args.num_refine <= 0:
            return
            
        # Functional map loss
        self.fmap_loss = (
            SURFMNetLoss(w_bij=1.0, w_orth=1.0, w_lap=0.0) 
            if not self.args.partial 
            else PartialFmapsLoss(w_bij=1.0, w_orth=1.0)
        )
        
        # Alignment loss
        self.align_loss = SquaredFrobeniusLoss(loss_weight=1.0)
        
        # Optimal transport loss
        self.ot_loss = SW(L=200, p=2, loss_weight=100., bidirectional=True)
        
        # Dirichlet loss weight
        if self.args.non_isometric:
            w_dirichlet = 5.0
        else:
            w_dirichlet = 1.0 if self.args.partial else 0.0
            
        self.dirichlet_loss = DirichletLoss(loss_weight=w_dirichlet)
    
    def compute_features(self, normalize: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute features for both shapes.
        
        Args:
            normalize: Whether to normalize features
            
        Returns:
            Tuple of features for source and target shapes
        """
        feat_x = self.feature_extractor(self.vert_x.unsqueeze(0), self.face_x.unsqueeze(0))
        feat_y = self.feature_extractor(self.vert_y.unsqueeze(0), self.face_y.unsqueeze(0))
        
        # Normalize features if requested
        if normalize:
            feat_x = F.normalize(feat_x, dim=-1, p=2)
            feat_y = F.normalize(feat_y, dim=-1, p=2)
            
        return feat_x, feat_y
    
    def compute_permutation_matrix(
        self, 
        feat_x: torch.Tensor, 
        feat_y: torch.Tensor, 
        bidirectional: bool = False, 
        normalize: bool = True
    ) -> torch.Tensor:
        """Compute permutation matrix between feature sets.
        
        Args:
            feat_x: Features from first shape
            feat_y: Features from second shape
            bidirectional: Whether to compute both Pxy and Pyx
            normalize: Whether to normalize features
            
        Returns:
            Permutation matrix or tuple of permutation matrices
        """
        # Normalize features if requested
        if normalize:
            feat_x = F.normalize(feat_x, dim=-1, p=2)
            feat_y = F.normalize(feat_y, dim=-1, p=2)
            
        # Compute similarity matrix
        similarity = torch.bmm(feat_x, feat_y.transpose(1, 2))
        
        # Apply Sinkhorn normalization
        Pxy = self.permutation(similarity)
        
        if bidirectional:
            Pyx = self.permutation(similarity.transpose(1, 2))
            return Pxy, Pyx
        else:
            return Pxy
    
    @staticmethod
    def dist_mat(x: torch.Tensor, y: torch.Tensor, inplace: bool = True) -> torch.Tensor:
        """Compute distance matrix between two sets of features.
        
        Args:
            x: First set of features
            y: Second set of features
            inplace: Whether to modify tensors in-place
            
        Returns:
            Distance matrix
        """
        d = torch.mm(x, y.transpose(0, 1))
        v_x = torch.sum(x ** 2, 1).unsqueeze(1)
        v_y = torch.sum(y ** 2, 1).unsqueeze(0)
        d *= -2
        
        if inplace:
            d += v_x
            d += v_y
        else:
            d = d + v_x
            d = d + v_y
            
        return d
    
    @staticmethod
    def sinkhorn(d: torch.Tensor, sigma: float = 0.1, num_sink: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply Sinkhorn algorithm to distance matrix.
        
        Args:
            d: Distance matrix
            sigma: Temperature parameter
            num_sink: Number of Sinkhorn iterations
            
        Returns:
            Tuple of transport plans (P, P_transpose)
        """
        d = d / d.mean()
        log_p = -d / (2 * sigma**2)
        
        for _ in range(num_sink):
            log_p = log_p - torch.logsumexp(log_p, dim=1, keepdim=True)
            log_p = log_p - torch.logsumexp(log_p, dim=0, keepdim=True)
            
        log_p = log_p - torch.logsumexp(log_p, dim=1, keepdim=True)
        p = torch.exp(log_p)
        
        log_p = log_p - torch.logsumexp(log_p, dim=0, keepdim=True)
        p_adj = torch.exp(log_p).transpose(0, 1)
        
        return p, p_adj
    
    def update_network(self, loss_metrics: Dict[str, torch.Tensor]) -> float:
        """Update network parameters based on loss metrics.
        
        Args:
            loss_metrics: Dictionary of loss values
            
        Returns:
            Total loss value
        """
        # Compute total loss
        loss = 0.0
        for k, v in loss_metrics.items():
            if k != 'l_total':
                loss += v
                
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Backward pass
        loss.backward()
        
        # Clip gradient for stability
        torch.nn.utils.clip_grad_norm_(self.feature_extractor.parameters(), 1.0)
        
        # Update weights
        self.optimizer.step()
        
        return loss
    
    def refine(self) -> None:
        """Perform adaptive refinement."""
        if self.args.num_refine <= 0:
            return
            
        print('Performing adaptive refinement...')
        
        # Set networks to refinement mode
        self.permutation.hard = False
        self.feature_extractor.train()
        
        # Refinement loop
        pbar = tqdm(range(self.args.num_refine))
        for _ in pbar:
            # Compute features
            feat_x, feat_y = self.compute_features()
            
            # Compute functional maps
            Cxy, Cyx = self.fmap_net(
                feat_x, feat_y, 
                self.evals_x.unsqueeze(0), self.evals_y.unsqueeze(0),
                self.evecs_trans_x.unsqueeze(0), self.evecs_trans_y.unsqueeze(0)
            )
            
            # Compute distance matrix and entropic Sinkhorn
            distance_matrix = self.dist_mat(feat_x.squeeze(0), feat_y.squeeze(0), False)
            Pxy, Pyx = self.sinkhorn(distance_matrix, sigma=0.1, num_sink=10)
            Pxy = Pxy.unsqueeze(0)
            Pyx = Pyx.unsqueeze(0)
            
            # Compute functional map regularization loss
            loss_metrics = self.fmap_loss(
                Cxy, Cyx, 
                self.evals_x.unsqueeze(0), self.evals_y.unsqueeze(0)
            )
            
            # Compute estimated functional map
            Cxy_est = torch.bmm(
                self.evecs_trans_y.unsqueeze(0), 
                torch.bmm(Pyx, self.evecs_x.unsqueeze(0))
            )
            
            # Compute alignment loss
            loss_metrics['l_align'] = self.align_loss(Cxy, Cxy_est)
            
            # Compute OT loss
            loss_metrics['l_ot'] = self.ot_loss(feat_x, feat_y, Pxy.detach(), Pyx.detach())
            
            # Add bidirectional alignment loss if not partial
            if not self.args.partial:
                Cyx_est = torch.bmm(
                    self.evecs_trans_x.unsqueeze(0), 
                    torch.bmm(Pxy, self.evecs_y.unsqueeze(0))
                )
                loss_metrics['l_align'] += self.align_loss(Cyx, Cyx_est)
            
            # Compute Dirichlet energy if non-isometric
            if self.args.non_isometric:
                loss_metrics['l_d'] = (
                    self.dirichlet_loss(
                        torch.bmm(Pxy, self.vert_y.unsqueeze(0)), 
                        self.Lx.to_dense().unsqueeze(0)
                    ) +
                    self.dirichlet_loss(
                        torch.bmm(Pyx, self.vert_x.unsqueeze(0)), 
                        self.Ly.to_dense().unsqueeze(0)
                    )
                )
            
            # Update network
            loss = self.update_network(loss_metrics)
            pbar.set_description(f'Total loss: {loss:.4f}')
    
    def compute_correspondence(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute correspondence between shapes.
        
        Returns:
            Tuple containing point-to-point map, permutation matrix, and functional map
        """
        # Set networks to evaluation mode
        self.feature_extractor.eval()
        self.permutation.hard = True
        
        # Compute features
        with torch.no_grad():
            feat_x, feat_y = self.compute_features(normalize=True)
        
        # Process based on isometry assumption
        if self.args.non_isometric:
            # Nearest neighbor query for non-isometric case
            p2p = nn_query(feat_x, feat_y).squeeze()
            
            # Compute functional map from point-to-point map
            Cxy = self.evecs_trans_y @ self.evecs_x[p2p]
            
            # Compute permutation matrix from functional map
            Pyx = self.evecs_y @ Cxy @ self.evecs_trans_x
        else:
            # Compute permutation matrix for isometric case
            Pyx = self.compute_permutation_matrix(
                feat_y, feat_x, bidirectional=False
            ).squeeze(0)
            
            # Compute functional map from permutation matrix
            Cxy = self.evecs_trans_y @ (Pyx @ self.evecs_x)
            
            # Convert functional map to point-to-point map
            p2p = fmap2pointmap(Cxy, self.evecs_x, self.evecs_y)
            
            # Recompute permutation matrix from functional map for consistency
            Pyx = self.evecs_y @ Cxy @ self.evecs_trans_x
        
        return p2p, Pyx, Cxy
    
    def save_results(self, p2p: torch.Tensor, Pyx: torch.Tensor, Cxy: torch.Tensor) -> None:
        """Save correspondence results.
        
        Args:
            p2p: Point-to-point map
            Pyx: Permutation matrix
            Cxy: Functional map
        """
        # Create result directory
        os.makedirs(self.args.result_path, exist_ok=True)
        
        # Get shape names
        name_x = os.path.splitext(os.path.basename(self.args.path_src_shape))[0]
        name_y = os.path.splitext(os.path.basename(self.args.path_tar_shape))[0]
        
        # Define output files
        file_x = os.path.join(self.args.result_path, f'{name_x}.obj')
        file_y = os.path.join(self.args.result_path, f'{name_x}-{name_y}.obj')
        
        # Convert to numpy
        Pyx_np = to_numpy(Pyx)
        
        # Save texture transfer result
        write_obj_pair(
            file_x, file_y, 
            self.vert_np_x, self.face_np_x, 
            self.vert_np_y, self.face_np_y, 
            Pyx_np, 'figures/texture.png'
        )
        
        # Save results for MATLAB if requested
        if self.args.save_mat:
            Cxy_np = to_numpy(Cxy)
            p2p_np = to_numpy(p2p)
            
            # Save functional map and point-wise correspondences
            save_dict = {'Cxy': Cxy_np, 'p2p': p2p_np + 1}  # plus one for MATLAB
            sio.savemat(os.path.join(self.args.result_path, f'{name_x}-{name_y}.mat'), save_dict)
        
        # Copy texture file
        os.makedirs(os.path.join(self.args.result_path, 'figures'), exist_ok=True)
        shutil.copy('figures/texture.png', os.path.join(self.args.result_path, 'figures/texture.png'))
        
        print(f'Finished, see the results under {self.args.result_path}')
    
    def run(self) -> None:
        """Run the complete adaptive refinement pipeline."""
        # Refinement phase
        start_refine = time.time()
        self.refine()
        end_refine = time.time()
        print(f'Finished refinement in {end_refine - start_refine:.4f} seconds')
        
        # Inference phase
        start_inference = time.time()
        p2p, Pyx, Cxy = self.compute_correspondence()
        end_inference = time.time()
        print(f'Finished inference in {end_inference - start_inference:.4f} seconds')
        
        # Save results
        self.save_results(p2p, Pyx, Cxy)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Adaptive refinement for shape correspondence')
    parser.add_argument('--path_src_shape', type=str, required=True, 
                        help='Path to the source shape')
    parser.add_argument('--path_tar_shape', type=str, required=True, 
                        help='Path to the target shape')
    parser.add_argument('--num_refine', type=int, default=15, 
                        help='Number of refinement iterations')
    parser.add_argument('--non_isometric', action='store_true', 
                        help='Use non-isometric correspondence')
    parser.add_argument('--partial', action='store_true', 
                        help='Use partial shape correspondence')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/faust.pth', 
                        help='Path to the checkpoint')
    parser.add_argument('--save_mat', action='store_true', 
                        help='Save results in MATLAB format')
    parser.add_argument('--result_path', type=str, default='refinement', 
                        help='Path to save the results')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(args)
    
    # Run adaptive refinement
    refiner = AdaptiveRefinement(args)
    refiner.run()
