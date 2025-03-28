import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional

from .base_model import BaseModel
from utils.registry import MODEL_REGISTRY
from utils.tensor_util import to_device
from utils.fmap_util import nn_query, fmap2pointmap
from utils.sinkhorn_util import sinkhorn_OT, dist_mat


@MODEL_REGISTRY.register()
class FMNetModel(BaseModel):
    """Functional Map Network Model for shape correspondence."""
    
    def __init__(self, opt: Dict[str, Any]):
        """Initialize FMNetModel.
        
        Args:
            opt: Dictionary containing model options
        """
        self.with_refine = opt.get('refine', -1)
        self.partial = opt.get('partial', False)
        self.non_isometric = opt.get('non-isometric', False)
        
        if self.with_refine > 0:
            opt['is_train'] = True
            
        super(FMNetModel, self).__init__(opt)

    def feed_data(self, data: Dict[str, Any]) -> None:
        """Process input data and compute losses.
        
        Args:
            data: Dictionary containing input data
        """
        # Get data pair
        data_x, data_y = to_device(data['first'], self.device), to_device(data['second'], self.device)

        # Extract features
        feat_x = self._extract_features(data_x)
        feat_y = self._extract_features(data_y)

        # Get spectral operators
        evals_x, evals_y = data_x['evals'], data_y['evals']
        evecs_x, evecs_y = data_x['evecs'], data_y['evecs']
        evecs_trans_x, evecs_trans_y = data_x['evecs_trans'], data_y['evecs_trans']

        # Compute functional maps
        Cxy, Cyx = self.networks['fmap_net'](feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y)

        # Compute surfmnet loss
        self.loss_metrics = self.losses['surfmnet_loss'](Cxy, Cyx, evals_x, evals_y)
        
        # Compute permutation matrices
        Pxy, Pyx = self._compute_permutation_matrices(feat_x, feat_y)

        # Compute alignment loss
        self._compute_alignment_loss(Cxy, Cyx, Pxy, Pyx, evecs_x, evecs_y, evecs_trans_x, evecs_trans_y)
        
        # Compute additional losses if available
        self._compute_additional_losses(data_x, data_y, feat_x, feat_y, Pxy, Pyx)

    def _extract_features(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract features from mesh data.
        
        Args:
            data: Dictionary containing mesh data
            
        Returns:
            Extracted features
        """
        return self.networks['feature_extractor'](data['verts'], data['faces'])

    def _compute_permutation_matrices(self, feat_x: torch.Tensor, feat_y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute permutation matrices between feature sets.
        
        Args:
            feat_x: Features from first shape
            feat_y: Features from second shape
            
        Returns:
            Tuple of permutation matrices (Pxy, Pyx)
        """
        if self.with_refine > 0:
            distance_matrix = dist_mat(feat_x.squeeze(0), feat_y.squeeze(0), False)
            Pxy, Pyx = sinkhorn_OT(distance_matrix, sigma=0.1, num_sink=10)
            return Pxy.unsqueeze(0), Pyx.unsqueeze(0)
        else:
            return self.compute_permutation_matrix(feat_x, feat_y, bidirectional=True)

    def _compute_alignment_loss(self, Cxy: torch.Tensor, Cyx: torch.Tensor, 
                               Pxy: torch.Tensor, Pyx: torch.Tensor,
                               evecs_x: torch.Tensor, evecs_y: torch.Tensor,
                               evecs_trans_x: torch.Tensor, evecs_trans_y: torch.Tensor) -> None:
        """Compute alignment loss between functional maps and permutation matrices.
        
        Args:
            Cxy, Cyx: Functional maps
            Pxy, Pyx: Permutation matrices
            evecs_x, evecs_y: Eigenvectors
            evecs_trans_x, evecs_trans_y: Transposed eigenvectors
        """
        # Compute estimated functional maps from permutation matrices
        Cxy_est = torch.bmm(evecs_trans_y, torch.bmm(Pyx, evecs_x))
        
        # Compute alignment loss
        self.loss_metrics['l_align'] = self.losses['align_loss'](Cxy, Cxy_est)
        
        # Add bidirectional loss if not partial
        if not self.partial:
            Cyx_est = torch.bmm(evecs_trans_x, torch.bmm(Pxy, evecs_y))
            self.loss_metrics['l_align'] += self.losses['align_loss'](Cyx, Cyx_est)

    def _compute_additional_losses(self, data_x: Dict[str, torch.Tensor], data_y: Dict[str, torch.Tensor],
                                  feat_x: torch.Tensor, feat_y: torch.Tensor,
                                  Pxy: torch.Tensor, Pyx: torch.Tensor) -> None:
        """Compute additional losses if available.
        
        Args:
            data_x, data_y: Input data dictionaries
            feat_x, feat_y: Feature tensors
            Pxy, Pyx: Permutation matrices
        """
        # Dirichlet loss if available
        if 'dirichlet_loss' in self.losses:
            Lx, Ly = data_x['L'], data_y['L']
            verts_x, verts_y = data_x['verts'], data_y['verts']
            self.loss_metrics['l_d'] = self.losses['dirichlet_loss'](torch.bmm(Pxy, verts_y), Lx) + \
                                      self.losses['dirichlet_loss'](torch.bmm(Pyx, verts_x), Ly)

        # Optimal transport loss if available
        if 'ot_loss' in self.losses:
            
            apply_after_ith_steps = self.opt['train']['losses']['ot_loss']['apply_after_ith_steps']
            if self.curr_iter > apply_after_ith_steps:
                self.loss_metrics['l_ot'] = self.losses['ot_loss'](feat_x, feat_y, Pxy, Pyx)
            # import pdb; pdb.set_trace()
            
        # Gromov-Wasserstein loss if available (experimental, not used in the paper)
        if 'gromov_loss' in self.losses:
            apply_after_ith_steps = self.opt['train']['losses']['gromov_loss']['apply_after_ith_steps']
            if self.curr_iter > apply_after_ith_steps:
                self.loss_metrics['l_gw'] = self.losses['gromov_loss'](
                    feat_x,
                    feat_y,
                    Pxy,
                    Pyx
                )
                

    def validate_single(self, data: Dict[str, Any], timer: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Validate on a single data pair.
        
        Args:
            data: Input data dictionary
            timer: Timer for benchmarking
            
        Returns:
            Tuple containing point-to-point map, permutation matrix, and functional map
        """
        # Get data pair
        data_x, data_y = to_device(data['first'], self.device), to_device(data['second'], self.device)

        # Save network state if refinement is enabled
        state_dict = None
        if self.with_refine > 0:
            state_dict = {'networks': self._get_networks_state_dict()}

        # Start timing
        timer.start()

        # Test-time refinement if enabled
        if self.with_refine > 0:
            self.refine(data)

        # Extract features
        feat_x = self.networks['feature_extractor'](data_x['verts'], data_x.get('faces'))
        feat_y = self.networks['feature_extractor'](data_y['verts'], data_y.get('faces'))

        # Get spectral operators
        evecs_x = data_x['evecs'].squeeze()
        evecs_y = data_y['evecs'].squeeze()
        evecs_trans_x = data_x['evecs_trans'].squeeze()
        evecs_trans_y = data_y['evecs_trans'].squeeze()

        # Process based on isometry assumption
        if self.non_isometric:
            p2p, Pyx, Cxy = self._process_non_isometric(
                feat_x, feat_y, evecs_x, evecs_y, evecs_trans_x, evecs_trans_y
            )
        else:
            p2p, Pyx, Cxy = self._process_isometric(
                feat_x, feat_y, evecs_x, evecs_y, evecs_trans_x, evecs_trans_y
            )

        # Stop timing
        timer.record()

        # Restore network state if refinement was used
        if self.with_refine > 0 and state_dict is not None:
            self.resume_model(state_dict, net_only=True, verbose=False)
            
        return p2p, Pyx, Cxy

    def _process_non_isometric(self, feat_x: torch.Tensor, feat_y: torch.Tensor,
                              evecs_x: torch.Tensor, evecs_y: torch.Tensor,
                              evecs_trans_x: torch.Tensor, evecs_trans_y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process non-isometric shape correspondence.
        
        Args:
            feat_x, feat_y: Feature tensors
            evecs_x, evecs_y: Eigenvectors
            evecs_trans_x, evecs_trans_y: Transposed eigenvectors
            
        Returns:
            Tuple of point-to-point map, permutation matrix, and functional map
        """
        # Normalize features
        feat_x = F.normalize(feat_x, dim=-1, p=2)
        feat_y = F.normalize(feat_y, dim=-1, p=2)

        # Nearest neighbor query
        p2p = nn_query(feat_x, feat_y).squeeze()

        # Compute functional map from point-to-point map
        Cxy = evecs_trans_y @ evecs_x[p2p]
        
        # Compute permutation matrix from functional map
        Pyx = evecs_y @ Cxy @ evecs_trans_x
        
        return p2p, Pyx, Cxy

    def _process_isometric(self, feat_x: torch.Tensor, feat_y: torch.Tensor,
                          evecs_x: torch.Tensor, evecs_y: torch.Tensor,
                          evecs_trans_x: torch.Tensor, evecs_trans_y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process isometric shape correspondence.
        
        Args:
            feat_x, feat_y: Feature tensors
            evecs_x, evecs_y: Eigenvectors
            evecs_trans_x, evecs_trans_y: Transposed eigenvectors
            
        Returns:
            Tuple of point-to-point map, permutation matrix, and functional map
        """
        # Compute permutation matrix
        Pyx = self.compute_permutation_matrix(feat_y, feat_x, bidirectional=False).squeeze()
        
        # Compute functional map from permutation matrix
        Cxy = evecs_trans_y @ (Pyx @ evecs_x)

        # Convert functional map to point-to-point map
        p2p = fmap2pointmap(Cxy, evecs_x, evecs_y)

        # Recompute permutation matrix from functional map for consistency
        Pyx = evecs_y @ Cxy @ evecs_trans_x
        
        return p2p, Pyx, Cxy

    def compute_permutation_matrix(self, feat_x: torch.Tensor, feat_y: torch.Tensor, 
                                  bidirectional: bool = False, normalize: bool = True) -> torch.Tensor:
        """Compute permutation matrix between feature sets.
        
        Args:
            feat_x: Features from first shape
            feat_y: Features from second shape
            bidirectional: Whether to compute both Pxy and Pyx
            normalize: Whether to normalize features
            
        Returns:
            Permutation matrix or tuple of permutation matrices
        """
        if normalize:
            feat_x = F.normalize(feat_x, dim=-1, p=2)
            feat_y = F.normalize(feat_y, dim=-1, p=2)
            
        # Compute similarity matrix
        similarity = torch.bmm(feat_x, feat_y.transpose(1, 2))

        # Apply Sinkhorn normalization
        Pxy = self.networks['permutation'](similarity)

        if bidirectional:
            Pyx = self.networks['permutation'](similarity.transpose(1, 2))
            return Pxy, Pyx
        else:
            return Pxy

    def refine(self, data: Dict[str, Any]) -> None:
        """Perform test-time refinement.
        
        Args:
            data: Input data dictionary
        """
        # Set network parameters for refinement
        self.networks['permutation'].hard = False
        self.networks['fmap_net'].bidirectional = True

        # Perform refinement iterations
        with torch.set_grad_enabled(True):
            for _ in range(self.with_refine):
                self.feed_data(data)
                self.optimize_parameters()

        # Reset network parameters
        self.networks['permutation'].hard = True
        self.networks['fmap_net'].bidirectional = False

    @torch.no_grad()
    def validation(self, dataloader: Any, tb_logger: Any, wandb: Optional[Any] = None, update: bool = True) -> None:
        """Perform validation on a dataset.
        
        Args:
            dataloader: Data loader for validation
            tb_logger: TensorBoard logger
            wandb: Weights & Biases logger
            update: Whether to update best model
        """
        # Set network parameters for validation
        self._set_validation_mode(True)
        
        # Call parent validation method
        super(FMNetModel, self).validation(dataloader, tb_logger, wandb=wandb, update=update)
        
        # Reset network parameters
        self._set_validation_mode(False)
    
    def _set_validation_mode(self, is_validation: bool) -> None:
        """Set network parameters for validation or training mode.
        
        Args:
            is_validation: Whether to set for validation
        """
        if 'permutation' in self.networks:
            self.networks['permutation'].hard = is_validation
            
        if 'fmap_net' in self.networks:
            self.networks['fmap_net'].bidirectional = not is_validation
