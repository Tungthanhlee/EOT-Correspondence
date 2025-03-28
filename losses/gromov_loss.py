#experimental, but you can try it out :)
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.functional import pad
from typing import Optional, Tuple, Any

from utils.registry import LOSS_REGISTRY

import time

class BaseGromovLoss(nn.Module):
    """Base class for Gromov-Wasserstein losses."""
    
    def __init__(
        self, 
        L: int = 100, 
        loss_weight: float = 1.0, 
        bidirectional: bool = False,
        apply_after_ith_steps: int = 0
    ):
        """Initialize base Gromov-Wasserstein loss.
        
        Args:
            L: Number of random projections
            loss_weight: Weight for the loss
            bidirectional: Whether to use bidirectional transport
            apply_after_ith_steps: Start applying loss after this many steps
        """
        super().__init__()
        self.loss_weight = loss_weight
        self.L = L
        self.bidirectional = bidirectional
        self.apply_after_ith_steps = apply_after_ith_steps
    
    def _generate_projections(
        self, 
        random_projection_dim: int,
        nproj: int,
        device: torch.device
    ) -> torch.Tensor:
        """Generate random projection directions.
        
        Args:
            random_projection_dim: Dimension of the projection space
            nproj: Number of projections
            device: Device to create tensors on
            
        Returns:
            Normalized random projections
        """
        P = torch.randn(random_projection_dim, nproj, device=device)
        return P / torch.sqrt(torch.sum(P**2, 0, True))


@LOSS_REGISTRY.register()
class SGW(BaseGromovLoss):
    """Sliced Gromov-Wasserstein distance."""
    
    def forward(
        self, 
        xs: torch.Tensor, 
        xt: torch.Tensor, 
        Pxy: Optional[torch.Tensor] = None, 
        Pyx: Optional[torch.Tensor] = None, 
        a: Optional[torch.Tensor] = None, 
        b: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute Sliced Gromov-Wasserstein distance.
        
        Args:
            xs: First point cloud features [B, N1, D]
            xt: Second point cloud features [B, N2, D]
            Pxy: Permutation matrix from xs to xt (not used in this implementation)
            Pyx: Permutation matrix from xt to xs (not used in this implementation)
            a: Weights for first distribution (not used in this implementation)
            b: Weights for second distribution (not used in this implementation)
            
        Returns:
            Sliced Gromov-Wasserstein distance
        """
        device = xs.device
        
        # Project the features to 1D
        xsp, xtp = self._sink(xs, xt, device, self.L)
        
        # Compute Gromov-Wasserstein distance
        d = self._gromov_1d(xsp, xtp)

        return self.loss_weight * d
    
    def _sink(
        self, 
        xs: torch.Tensor, 
        xt: torch.Tensor, 
        device: torch.device,
        nproj: int = 200, 
        P: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sink the points of the measure in the lowest dimension onto the highest dimension and apply projections.
        
        Args:
            xs: Source samples [B, N, D1]
            xt: Target samples [B, N, D2]
            device: Torch device
            nproj: Number of projections (ignored if P is not None)
            P: Projection matrix (optional)
            
        Returns:
            Projected source and target samples
        """
        dim_d = xs.shape[1]
        dim_p = xt.shape[1]
        
        # Pad the lower dimensional features to match the higher dimension
        if dim_d < dim_p:
            random_projection_dim = dim_p
            xs2 = torch.cat((xs, torch.zeros((xs.shape[0], dim_p-dim_d), device=device)), dim=1)
            xt2 = xt
        else:
            random_projection_dim = dim_d
            # import pdb; pdb.set_trace()
            xt2 = torch.cat((xt, torch.zeros((xt.shape[0], dim_d-dim_p), device=device)), dim=1)
            xs2 = xs
        
        # Generate or use provided projection matrix
        if P is None:
            p = self._generate_projections(random_projection_dim, nproj, device)
        else:
            p = P / torch.sqrt(torch.sum(P**2, 0, True))
        
        # Project the features
        try:
            xsp = torch.matmul(xs2, p)
            xtp = torch.matmul(xt2, p)
        except RuntimeError as error:
            print('----------------------------------------')
            print('xs original dim:', xs.shape)
            print('xt original dim:', xt.shape)
            print('dim_p:', dim_p)
            print('dim_d:', dim_d)
            print('random_projection_dim:', random_projection_dim)
            print('projector dimension:', p.shape)
            print('xs2 dim:', xs2.shape)
            print('xt2 dim:', xt2.shape)
            print('----------------------------------------')
            print(error)
            raise RuntimeError("Projection failed due to dimension mismatch")
        
        return xsp, xtp
    
    def _gromov_1d(
        self, 
        xs: torch.Tensor, 
        xt: torch.Tensor, 
        tolog: bool = False
    ) -> torch.Tensor:
        """Solve the Gromov-Wasserstein problem in 1D for each projection.
        
        Args:
            xs: Projected source samples [N, n_proj]
            xt: Projected target samples [N, n_proj]
            tolog: Whether to return timing information
            
        Returns:
            The SGW cost
        """
        if tolog:
            log = {}
            start_time = time.time()
        
        # Sort source samples
        xs2, _ = torch.sort(xs, dim=0)
        
        # Sort target samples in ascending and descending order
        xt_asc, _ = torch.sort(xt, dim=0)
        xt_desc, _ = torch.sort(xt, dim=0, descending=True)
        
        # Compute cost for both orderings
        if tolog:
            l1, t1 = self._compute_cost(xs2, xt_asc, tolog=tolog)
            l2, t2 = self._compute_cost(xs2, xt_desc, tolog=tolog)
        else:
            l1 = self._compute_cost(xs2, xt_asc, tolog=tolog)
            l2 = self._compute_cost(xs2, xt_desc, tolog=tolog)
        
        # Take the minimum cost between the two orderings
        result = torch.mean(torch.min(l1, l2))
        
        if tolog:
            end_time = time.time()
            log['g1d'] = end_time - start_time
            log['t1'] = t1
            log['t2'] = t2
            return result, log
        else:
            return result
    
    def _compute_cost(
        self, 
        xsp: torch.Tensor, 
        xtp: torch.Tensor, 
        tolog: bool = False
    ) -> torch.Tensor:
        """Compute the Gromov-Wasserstein cost.
        
        Args:
            xsp: 1D sorted samples for source
            xtp: 1D sorted samples for target
            tolog: Whether to return timing information
            
        Returns:
            Cost for each projection
        """
        if tolog:
            start_time = time.time()
        
        # Compute powers of the sorted samples
        xs = xsp
        xt = xtp
        
        xs2 = xs * xs
        xs3 = xs2 * xs
        xs4 = xs2 * xs2
        
        xt2 = xt * xt
        xt3 = xt2 * xt
        xt4 = xt2 * xt2
        
        # Compute sums
        X = torch.sum(xs, 0)
        X2 = torch.sum(xs2, 0)
        X3 = torch.sum(xs3, 0)
        X4 = torch.sum(xs4, 0)
        
        Y = torch.sum(xt, 0)
        Y2 = torch.sum(xt2, 0)
        Y3 = torch.sum(xt3, 0)
        Y4 = torch.sum(xt4, 0)
        
        # Compute cross terms
        xxyy_ = torch.sum((xs2) * (xt2), 0)
        xxy_ = torch.sum((xs2) * (xt), 0)
        xyy_ = torch.sum((xs) * (xt2), 0)
        xy_ = torch.sum((xs) * (xt), 0)
        
        n = xs.shape[0]
        
        # Compute the cost
        C2 = 2 * X2 * Y2 + 2 * (n * xxyy_ - 2 * Y * xxy_ - 2 * X * xyy_ + 2 * xy_ * xy_)
        
        power4_x = 2 * n * X4 - 8 * X3 * X + 6 * X2 * X2
        power4_y = 2 * n * Y4 - 8 * Y3 * Y + 6 * Y2 * Y2
        
        C = (1 / (n**2)) * (power4_x + power4_y - 2 * C2)
        
        if tolog:
            end_time = time.time()
            return C, end_time - start_time
        else:
            return C


@LOSS_REGISTRY.register()
class BiSGW(SGW):
    """Bidirectional Sliced Gromov-Wasserstein distance."""
    
    def forward(
        self, 
        xs: torch.Tensor, 
        xt: torch.Tensor, 
        Pxy: Optional[torch.Tensor] = None, 
        Pyx: Optional[torch.Tensor] = None, 
        a: Optional[torch.Tensor] = None, 
        b: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute Bidirectional Sliced Gromov-Wasserstein distance.
        
        Args:
            xs: First point cloud features [B, N1, D]
            xt: Second point cloud features [B, N2, D]
            Pxy: Permutation matrix from xs to xt
            Pyx: Permutation matrix from xt to xs
            a: Weights for first distribution (not used)
            b: Weights for second distribution (not used)
            
        Returns:
            Bidirectional Sliced Gromov-Wasserstein distance
        """
        device = xs.device
        
        # Forward direction
        xsp, xtp = self._sink(xs, xt, device, self.L)
        d_forward = self._gromov_1d(xsp, xtp)
        
        # Backward direction
        if self.bidirectional and Pxy is not None and Pyx is not None:
            # Apply permutation matrices
            xs_hat = torch.bmm(Pyx, xs)  # B x N2 x D
            xt_hat = torch.bmm(Pxy, xt)  # B x N1 x D
            
            # Project transformed features
            xs_hat_p, xt_hat_p = self._sink(xs_hat, xt_hat, device, self.L)
            d_backward = self._gromov_1d(xs_hat_p, xt_hat_p)
            
            # Combine both directions
            return self.loss_weight * (d_forward + d_backward) / 2
        
        return self.loss_weight * d_forward


def _cost(xsp,xtp,tolog=False):   
    """ Returns the GM cost eq (3) in [1]
    Parameters
    ----------
    xsp : tensor, shape (n, n_proj)
         1D sorted samples (after finding sigma opt) for each proj in the source
    xtp : tensor, shape (n, n_proj)
         1D sorted samples (after finding sigma opt) for each proj in the target
    tolog : bool
            Wether to return timings or not
    Returns
    -------
    C : tensor, shape (n_proj,1)
           Cost for each projection
    References
    ----------
    .. [1] Vayer Titouan, Chapel Laetitia, Flamary R{\'e}mi, Tavenard Romain
          and Courty Nicolas
          "Sliced Gromov-Wasserstein"
    """
    st=time.time()

    xs=xsp
    xt=xtp

    xs2=xs*xs
    xs3=xs2*xs
    xs4=xs2*xs2

    xt2=xt*xt
    xt3=xt2*xt
    xt4=xt2*xt2

    X=torch.sum(xs,0)
    X2=torch.sum(xs2,0)
    X3=torch.sum(xs3,0)
    X4=torch.sum(xs4,0)
    
    Y=torch.sum(xt,0)
    Y2=torch.sum(xt2,0)
    Y3=torch.sum(xt3,0)
    Y4=torch.sum(xt4,0)
    
    xxyy_=torch.sum((xs2)*(xt2),0)
    xxy_=torch.sum((xs2)*(xt),0)
    xyy_=torch.sum((xs)*(xt2),0)
    xy_=torch.sum((xs)*(xt),0)
    
            
    n=xs.shape[0]

    C2=2*X2*Y2+2*(n*xxyy_-2*Y*xxy_-2*X*xyy_+2*xy_*xy_)

    power4_x=2*n*X4-8*X3*X+6*X2*X2
    power4_y=2*n*Y4-8*Y3*Y+6*Y2*Y2

    C=(1/(n**2))*(power4_x+power4_y-2*C2)
        
        
    ed=time.time()
    
    if not tolog:
        return C 
    else:
        return C,ed-st


def gromov_1d(xs,xt,tolog=False): 
    """ Solves the Gromov in 1D (eq (2) in [1] for each proj
    Parameters
    ----------
    xsp : tensor, shape (n, n_proj)
         1D sorted samples for each proj in the source
    xtp : tensor, shape (n, n_proj)
         1D sorted samples for each proj in the target
    tolog : bool
            Wether to return timings or not
    fast: use the O(nlog(n)) cost or not
    Returns
    -------
    toreturn : tensor, shape (n_proj,1)
           The SGW cost for each proj
    References
    ----------
    .. [1] Vayer Titouan, Chapel Laetitia, Flamary R{\'e}mi, Tavenard Romain
          and Courty Nicolas
          "Sliced Gromov-Wasserstein"
    """
    
    if tolog:
        log={}
    
    st=time.time()
    xs2,i_s=torch.sort(xs,dim=0)
    
    if tolog:
        xt_asc,i_t=torch.sort(xt,dim=0) #sort increase
        xt_desc,i_t=torch.sort(xt,dim=0,descending=True) #sort deacrese
        l1,t1=_cost(xs2,xt_asc,tolog=tolog)
        l2,t2=_cost(xs2,xt_desc,tolog=tolog)
    else:
        xt_asc,i_t=torch.sort(xt,dim=0)
        xt_desc,i_t=torch.sort(xt,dim=0,descending=True)
        l1=_cost(xs2,xt_asc,tolog=tolog)
        l2=_cost(xs2,xt_desc,tolog=tolog)   
    toreturn=torch.mean(torch.min(l1,l2)) 
    ed=time.time()  
   
    if tolog:
        log['g1d']=ed-st
        log['t1']=t1
        log['t2']=t2
 
    if tolog:
        return toreturn,log
    else:
        return toreturn

def sink_(xs,xt,device,nproj=200,P=None): #Delta operator (here just padding)
    """ Sinks the points of the measure in the lowest dimension onto the highest dimension and applies the projections.
    Only implemented with the 0 padding Delta=Delta_pad operator (see [1])
    Parameters
    ----------
    xs : tensor, shape (n, p)
         Source samples
    xt : tensor, shape (n, q)
         Target samples
    device :  torch device
    nproj : integer
            Number of projections. Ignored if P is not None
    P : tensor, shape (max(p,q),n_proj)
        Projection matrix
    Returns
    -------
    xsp : tensor, shape (n,n_proj)
           Projected source samples 
    xtp : tensor, shape (n,n_proj)
           Projected target samples 
    References
    ----------
    .. [1] Vayer Titouan, Chapel Laetitia, Flamary R{\'e}mi, Tavenard Romain
          and Courty Nicolas
          "Sliced Gromov-Wasserstein"
    """  
    dim_d= xs.shape[1]
    dim_p= xt.shape[1]
    
    if dim_d<dim_p:
        random_projection_dim = dim_p
        xs2=torch.cat((xs,torch.zeros((xs.shape[0],dim_p-dim_d)).to(device)),dim=1)
        xt2=xt
    else:
        random_projection_dim = dim_d
        xt2=torch.cat((xt,torch.zeros((xt.shape[0],dim_d-dim_p)).to(device)),dim=1)
        xs2=xs
     
    if P is None:
        P=torch.randn(random_projection_dim,nproj)
    p=P/torch.sqrt(torch.sum(P**2,0,True))
    
    try:
    
        xsp=torch.matmul(xs2,p.to(device))
        xtp=torch.matmul(xt2,p.to(device))
    except RuntimeError as error:
        print('----------------------------------------')
        print('xs origi dim :', xs.shape)
        print('xt origi dim :', xt.shape)
        print('dim_p :', dim_p)
        print('dim_d :', dim_d)
        print('random_projection_dim : ',random_projection_dim)
        print('projector dimension : ',p.shape)
        print('xs2 dim :', xs2.shape)
        print('xt2 dim :', xt2.shape)
        print('xs_tmp dim :', xs2.shape)
        print('xt_tmp dim :', xt2.shape)
        print('----------------------------------------')
        print(error)
        raise BadShapeError
    
    return xsp,xtp


# def distance_tensor_batch(pts_src: torch.Tensor, pts_dst: torch.Tensor, p: int = 2):
#     """
#     Returns the matrix of ||x_i-y_j||_p^p.
#     :param pts_src: [B, R, D] matrix
#     :param pts_dst: [B, C, D] matrix
#     :param p:
#     :return: [B, R, C, D] distance matrix
#     """
#     x_col = pts_src.unsqueeze(2)
#     y_row = pts_dst.unsqueeze(1)
#     print(x_col.shape, y_row.shape)
#     distance = torch.abs(x_col - y_row) ** p
#     return distance




# @LOSS_REGISTRY.register()
# class SGW(nn.Module):
#     def __init__(self,L=100, p=2, loss_weight=1.0):
#         """
#         Init Sliced Gromov Wasserstein distance.
#         Args:
#             L (int): Number of random projections.
#             p (int): p-Wasserstein distance.
#             loss_weight (float): Loss weight.
#         """
        
#         super().__init__()
#         self.loss_weight = loss_weight
#         self.L = L
#         self.p = p
        
#     def forward(self, X, Y):
#         """
#         Compute the Sliced Gromov Wasserstein Distance between two probability measures. Only support equal supports
#         Args:
#             X (torch.Tensor): Measure 1. Shape: [B, N, D]
#             Y (torch.Tensor): Measure 2. Shape: [B, N, D]
            
#         """
#         assert X.shape[0] == Y.shape[0], "Batch size must be equal"
#         assert X.shape[1] == Y.shape[1], "Number of supports must be equal"
        
#         dim = X.shape[2]
#         num_marginal = X.shape[0]
        
#         X = X.view(X.shape[0], X.shape[1], -1)
#         Y = Y.view(Y.shape[0], Y.shape[1], -1)
        
#         # Thetas = B x L x D
#         thetas = torch.randn(num_marginal, dim, self.L, device=X.device)
#         thetas = thetas/torch.sqrt(torch.sum(thetas**2, dim=2, keepdim=True)) # normalize
        
#         # Project X and Y onto the thetas
#         X_proj = torch.matmul(X, thetas) # B x N x L
#         Y_proj = torch.matmul(Y, thetas) # B x N x L
        
#         # sort both measures
#         X_proj = torch.sort(X_proj, dim=1)[0]
#         Y_proj1 = torch.sort(Y_proj, dim=1)[0]
#         Y_proj2 = torch.sort(Y_proj, dim=1, descending=True)[0]
        
#         # compute the distance between the sorted measures
#         X_diff = distance_tensor_batch(X_proj, X_proj, p=self.p)
#         Y_diff1 = distance_tensor_batch(Y_proj1, Y_proj1, p=self.p)
#         Y_diff2 = distance_tensor_batch(Y_proj2, Y_proj2, p=self.p)
        
#         out1 = torch.sum(torch.sum((X_diff - Y_diff1) ** p, dim=1), dim=2)
#         out2 = torch.sum(torch.sum((X_diff - Y_diff2) ** p, dim=1), dim=2)
#         one_dimensional_SGW =  torch.sum(torch.min(out1, out2))
        
#         return self.loss_weight*torch.mean(one_dimensional_SGW)