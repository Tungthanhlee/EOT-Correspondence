import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.functional import pad
from typing import Optional, Tuple, Any

from utils.registry import LOSS_REGISTRY


def one_dimensional_wasserstein_parallel(
    u_values: torch.Tensor, 
    v_values: torch.Tensor, 
    u_weights: Optional[torch.Tensor] = None, 
    v_weights: Optional[torch.Tensor] = None, 
    p: int = 2
) -> torch.Tensor:
    """Compute 1D Wasserstein distance between batches of distributions.
    
    Args:
        u_values: First distribution values [B, N1, L]
        v_values: Second distribution values [B, N2, L]
        u_weights: Weights for first distribution (optional)
        v_weights: Weights for second distribution (optional)
        p: Power for Wasserstein distance
        
    Returns:
        Wasserstein distances [B]
    """
    n = u_values.shape[1]
    m = v_values.shape[1]

    # Initialize uniform weights if not provided
    if u_weights is None:
        u_weights = torch.full(u_values.shape, 1. / n,
                              dtype=u_values.dtype, device=u_values.device)
    elif u_weights.ndim != u_values.ndim:
        u_weights = torch.repeat_interleave(
            u_weights[..., None], u_values.shape[-1], -1)
            
    if v_weights is None:
        v_weights = torch.full(v_values.shape, 1. / m,
                              dtype=v_values.dtype, device=v_values.device)
    elif v_weights.ndim != v_values.ndim:
        v_weights = torch.repeat_interleave(
            v_weights[..., None], v_values.shape[-1], -1)

    # Sort values
    u_sorter = torch.sort(u_values, 1)[1]
    u_values = torch.gather(u_values, 1, u_sorter)
    u_weights = torch.gather(u_weights, 1, u_sorter)

    v_sorter = torch.sort(v_values, 1)[1]
    v_values = torch.gather(v_values, 1, v_sorter)
    v_weights = torch.gather(v_weights, 1, v_sorter)

    # Compute cumulative weights
    u_cumweights = torch.cumsum(u_weights, 1)
    v_cumweights = torch.cumsum(v_weights, 1)

    # Quantile computation
    qs = torch.sort(torch.cat((u_cumweights, v_cumweights), 1), 1)[0]
    u_quantiles = quantile_function_parallel(qs, u_cumweights, u_values)
    v_quantiles = quantile_function_parallel(qs, v_cumweights, v_values)

    # Compute Wasserstein distance
    how_pad = (0, 0, 1, 0)
    qs = pad(qs, how_pad)
    delta = qs[:, 1:, :] - qs[:, :-1, :]
    diff_quantiles = torch.abs(u_quantiles - v_quantiles)
    
    return torch.mean(torch.sum(delta * torch.pow(diff_quantiles, p), dim=1), dim=1)**(1./p)


def quantile_function_parallel(
    qs: torch.Tensor, 
    cws: torch.Tensor, 
    xs: torch.Tensor
) -> torch.Tensor:
    """Compute quantile function for parallel processing.
    
    Args:
        qs: Quantile points
        cws: Cumulative weights
        xs: Distribution values
        
    Returns:
        Quantile values
    """
    n = xs.shape[1]
    cws = torch.permute(cws, (0, 2, 1)).contiguous()
    qs = torch.permute(qs, (0, 2, 1)).contiguous()
    idx = torch.permute(torch.searchsorted(cws, qs, right=False), (0, 2, 1))
    return torch.gather(xs, 1, torch.clamp(idx, 0, n - 1))


def one_dimensional_wasserstein(
    u_values: torch.Tensor, 
    v_values: torch.Tensor, 
    u_weights: Optional[torch.Tensor] = None, 
    v_weights: Optional[torch.Tensor] = None, 
    p: int = 2
) -> torch.Tensor:
    """Compute 1D Wasserstein distance between distributions.
    
    Similar to one_dimensional_wasserstein_parallel but returns distances per projection.
    
    Args:
        u_values: First distribution values [B, N1, L]
        v_values: Second distribution values [B, N2, L]
        u_weights: Weights for first distribution (optional)
        v_weights: Weights for second distribution (optional)
        p: Power for Wasserstein distance
        
    Returns:
        Wasserstein distances per projection [B, L]
    """
    n = u_values.shape[1]
    m = v_values.shape[1]

    # Initialize uniform weights if not provided
    if u_weights is None:
        u_weights = torch.full(u_values.shape, 1. / n,
                              dtype=u_values.dtype, device=u_values.device)
    elif u_weights.ndim != u_values.ndim:
        u_weights = torch.repeat_interleave(
            u_weights[..., None], u_values.shape[-1], -1)
            
    if v_weights is None:
        v_weights = torch.full(v_values.shape, 1. / m,
                              dtype=v_values.dtype, device=v_values.device)
    elif v_weights.ndim != v_values.ndim:
        v_weights = torch.repeat_interleave(
            v_weights[..., None], v_values.shape[-1], -1)

    # Sort values
    u_sorter = torch.sort(u_values, 1)[1]
    u_values = torch.gather(u_values, 1, u_sorter)
    u_weights = torch.gather(u_weights, 1, u_sorter)

    v_sorter = torch.sort(v_values, 1)[1]
    v_values = torch.gather(v_values, 1, v_sorter)
    v_weights = torch.gather(v_weights, 1, v_sorter)

    # Compute cumulative weights
    u_cumweights = torch.cumsum(u_weights, 1)
    v_cumweights = torch.cumsum(v_weights, 1)

    # Quantile computation
    qs = torch.sort(torch.cat((u_cumweights, v_cumweights), 1), 1)[0]
    u_quantiles = quantile_function_parallel(qs, u_cumweights, u_values)
    v_quantiles = quantile_function_parallel(qs, v_cumweights, v_values)

    # Compute Wasserstein distance
    how_pad = (0, 0, 1, 0)
    qs = pad(qs, how_pad)
    delta = qs[:, 1:, :] - qs[:, :-1, :]
    diff_quantiles = torch.abs(u_quantiles - v_quantiles)
    
    return torch.sum(delta * torch.pow(diff_quantiles, p), dim=1)


class BaseOTLoss(nn.Module):
    """Base class for optimal transport losses."""
    
    def __init__(
        self, 
        L: int = 100, 
        p: int = 2, 
        loss_weight: float = 1.0, 
        bidirectional: bool = False,
        apply_after_ith_steps: int = 0
    ):
        """Initialize base OT loss.
        
        Args:
            L: Number of random projections
            p: Power for Wasserstein distance
            loss_weight: Weight for the loss
            bidirectional: Whether to use bidirectional transport
        """
        super().__init__()
        self.loss_weight = loss_weight
        self.L = L
        self.p = p
        self.bidirectional = bidirectional
    
    def _generate_projections(
        self, 
        batch_size: int, 
        dim: int, 
        device: torch.device
    ) -> torch.Tensor:
        """Generate random projection directions.
        
        Args:
            batch_size: Number of batches
            dim: Dimension of the feature space
            device: Device to create tensors on
            
        Returns:
            Normalized random projections [B, L, D]
        """
        thetas = torch.randn(batch_size, self.L, dim, device=device)
        return thetas / torch.sqrt(torch.sum(thetas**2, dim=2, keepdim=True))
    
    def _project_features(
        self, 
        features: torch.Tensor, 
        projections: torch.Tensor
    ) -> torch.Tensor:
        """Project features onto random directions.
        
        Args:
            features: Feature vectors [B, N, D]
            projections: Projection directions [B, L, D]
            
        Returns:
            Projected features [B, N, L]
        """
        return torch.matmul(features, projections.transpose(1, 2))


@LOSS_REGISTRY.register()
class SWdirect(BaseOTLoss):
    """Direct Sliced Wasserstein distance without permutation matrices."""
    
    def forward(
        self, 
        X: torch.Tensor, 
        Y: torch.Tensor, 
        Pxy: torch.Tensor, 
        Pyx: torch.Tensor, 
        a: Optional[torch.Tensor] = None, 
        b: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute direct Sliced Wasserstein distance.
        
        Args:
            X: First point cloud features [B, N1, D]
            Y: Second point cloud features [B, N2, D]
            Pxy, Pyx: Not used in this implementation
            a: Weights for first distribution (optional)
            b: Weights for second distribution (optional)
            
        Returns:
            Sliced Wasserstein distance
        """
        batch_size = X.shape[0]
        dim = X.shape[2]
        
        # Reshape inputs if needed
        X = X.view(X.shape[0], X.shape[1], -1)
        Y = Y.view(Y.shape[0], Y.shape[1], -1)
        
        # Generate random projections
        thetas = self._generate_projections(batch_size, dim, X.device)
        
        # Project features
        X_proj = self._project_features(X, thetas)
        Y_proj = self._project_features(Y, thetas)
        
        # Compute Wasserstein distance
        return self.loss_weight * torch.mean(
            one_dimensional_wasserstein_parallel(X_proj, Y_proj, a, b, self.p)
        )


@LOSS_REGISTRY.register()
class SW(BaseOTLoss):
    """Sliced Wasserstein distance with permutation matrices."""
    
    def forward(
        self, 
        X: torch.Tensor, 
        Y: torch.Tensor, 
        Pxy: torch.Tensor, 
        Pyx: torch.Tensor, 
        a: Optional[torch.Tensor] = None, 
        b: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute Sliced Wasserstein distance with permutation.
        
        Args:
            X: First point cloud features [B, N1, D]
            Y: Second point cloud features [B, N2, D]
            Pxy: Permutation matrix from X to Y [B, N1, N2]
            Pyx: Permutation matrix from Y to X [B, N2, N1]
            a: Weights for first distribution (optional)
            b: Weights for second distribution (optional)
            
        Returns:
            Sliced Wasserstein distance
        """
        batch_size = X.shape[0]
        dim = X.shape[2]
        
        # Reshape inputs if needed
        X = X.view(X.shape[0], X.shape[1], -1)
        Y = Y.view(Y.shape[0], Y.shape[1], -1)
        
        # Apply permutation matrices
        X_hat = torch.bmm(Pyx, X)  # B x N2 x D
        Y_hat = torch.bmm(Pxy, Y)  # B x N1 x D
        
        # Generate random projections
        thetas = self._generate_projections(batch_size, dim, X.device)
        
        # Project features
        X_proj = self._project_features(X, thetas)
        Y_proj = self._project_features(Y, thetas)
        
        if self.bidirectional:
            # Project transformed features
            X_hat_proj = self._project_features(X_hat, thetas)
            Y_hat_proj = self._project_features(Y_hat, thetas)
            
            # Compute bidirectional Wasserstein distance
            swd = torch.mean(
                one_dimensional_wasserstein_parallel(X_proj, Y_hat_proj, a, b, self.p) + 
                one_dimensional_wasserstein_parallel(X_hat_proj, Y_proj, a, b, self.p)
            )
        else:
            # Project only Y_hat for unidirectional distance
            Y_hat_proj = self._project_features(Y_hat, thetas)
            swd = torch.mean(
                one_dimensional_wasserstein_parallel(X_proj, Y_hat_proj, a, b, self.p)
            )
            
        return self.loss_weight * swd


@LOSS_REGISTRY.register()
class ISEBSW(BaseOTLoss):
    """Importance Sampling Enhanced Bidirectional Sliced Wasserstein distance."""
    
    def forward(
        self, 
        X: torch.Tensor, 
        Y: torch.Tensor, 
        Pxy: torch.Tensor, 
        Pyx: torch.Tensor, 
        a: Optional[torch.Tensor] = None, 
        b: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute ISEBSW distance.
        
        Args:
            X: First point cloud features [B, N1, D]
            Y: Second point cloud features [B, N2, D]
            Pxy: Permutation matrix from X to Y [B, N1, N2]
            Pyx: Permutation matrix from Y to X [B, N2, N1]
            a: Weights for first distribution (optional)
            b: Weights for second distribution (optional)
            
        Returns:
            ISEBSW distance
        """
        batch_size = X.shape[0]
        dim = X.shape[2]
        
        # Reshape inputs if needed
        X = X.view(X.shape[0], X.shape[1], -1)
        Y = Y.view(Y.shape[0], Y.shape[1], -1)
        
        # Apply permutation matrices
        X_hat = torch.bmm(Pyx, X)  # B x N2 x D
        Y_hat = torch.bmm(Pxy, Y)  # B x N1 x D
        
        # Generate random projections
        thetas = self._generate_projections(batch_size, dim, X.device)
        
        # Project features
        X_proj = self._project_features(X, thetas)
        Y_proj = self._project_features(Y, thetas)
        
        if self.bidirectional:
            # Project transformed features
            X_hat_proj = self._project_features(X_hat, thetas)
            Y_hat_proj = self._project_features(Y_hat, thetas)
            
            # Compute Wasserstein distances for each projection
            wasserstein_distance1 = one_dimensional_wasserstein(X_proj, Y_hat_proj, a, b, self.p)
            wasserstein_distance2 = one_dimensional_wasserstein(X_hat_proj, Y_proj, a, b, self.p)
            
            # Reshape for importance weighting
            wasserstein_distance1 = wasserstein_distance1.view(batch_size, 1, self.L)
            wasserstein_distance2 = wasserstein_distance2.view(batch_size, 1, self.L)
            
            # Combine distances
            wasserstein_distance = wasserstein_distance1 + wasserstein_distance2
            
            # Apply importance weighting
            weights = torch.softmax(wasserstein_distance, dim=2)
            swd = torch.sum(weights * wasserstein_distance, dim=2).mean()
        else:
            # Project only Y_hat for unidirectional distance
            Y_hat_proj = self._project_features(Y_hat, thetas)
            
            # Compute Wasserstein distances for each projection
            wasserstein_distance = one_dimensional_wasserstein(X_proj, Y_hat_proj, a, b, self.p)
            wasserstein_distance = wasserstein_distance.view(batch_size, 1, self.L)
            
            # Apply importance weighting
            weights = torch.softmax(wasserstein_distance, dim=2)
            swd = torch.sum(weights * wasserstein_distance, dim=2).mean()
            
        return self.loss_weight * torch.pow(swd, 1./self.p)


@LOSS_REGISTRY.register()
class MaxSW(BaseOTLoss):
    """Max-Sliced Wasserstein distance."""
    
    def __init__(
        self, 
        L: int = 100, 
        p: int = 2, 
        s_lr: float = 1e-2, 
        T: int = 100, 
        loss_weight: float = 1.0, 
        bidirectional: bool = False
    ):
        """Initialize Max-Sliced Wasserstein distance.
        
        Args:
            L: Number of random projections
            p: Power for Wasserstein distance
            s_lr: Learning rate for inner optimization
            T: Number of steps for inner optimization
            loss_weight: Weight for the loss
            bidirectional: Whether to use bidirectional transport
        """
        super().__init__(L, p, loss_weight, bidirectional)
        self.s_lr = s_lr
        self.T = T
    
    def forward(
        self, 
        X: torch.Tensor, 
        Y: torch.Tensor, 
        Pxy: torch.Tensor, 
        Pyx: torch.Tensor, 
        a: Optional[torch.Tensor] = None, 
        b: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute Max-Sliced Wasserstein distance.
        
        Args:
            X: First point cloud features [B, N1, D]
            Y: Second point cloud features [B, N2, D]
            Pxy: Permutation matrix from X to Y [B, N1, N2]
            Pyx: Permutation matrix from Y to X [B, N2, N1]
            a: Weights for first distribution (optional)
            b: Weights for second distribution (optional)
            
        Returns:
            Max-Sliced Wasserstein distance
        """
        batch_size = X.shape[0]
        dim = X.shape[2]
        
        # Reshape inputs if needed
        X = X.view(X.shape[0], X.shape[1], -1)
        Y = Y.view(Y.shape[0], Y.shape[1], -1)
        
        # Initialize projection direction (one per batch)
        thetas = torch.randn(batch_size, 1, dim, device=X.device)
        thetas = thetas / torch.sqrt(torch.sum(thetas**2, dim=2, keepdim=True))
        
        # Make projection direction trainable
        thetas = Variable(thetas, requires_grad=True)
        X_detach = X.detach()
        Y_detach = Y.detach()
        
        # Create optimizer for projection direction
        optimizer = torch.optim.SGD([thetas], lr=self.s_lr)
        
        if not self.bidirectional:
            # Optimize projection direction to maximize Wasserstein distance
            for _ in range(self.T-1):
                X_proj = torch.matmul(X_detach, thetas.transpose(1, 2))
                Y_proj = torch.matmul(Y_detach, thetas.transpose(1, 2))
                
                # Negative distance for maximization
                negative_distance = -torch.mean(
                    one_dimensional_wasserstein_parallel(X_proj, Y_proj, a, b, self.p)
                )
                
                # Update projection direction
                optimizer.zero_grad()
                negative_distance.backward()
                optimizer.step()
                
                # Normalize projection direction
                thetas.data = thetas.data / torch.sqrt(
                    torch.sum(thetas.data ** 2, dim=2, keepdim=True)
                )
            
            # Compute final distance with optimized projection
            X_proj = torch.matmul(X, thetas.transpose(1, 2))
            Y_proj = torch.matmul(Y, thetas.transpose(1, 2))
            swd = torch.mean(
                one_dimensional_wasserstein_parallel(X_proj, Y_proj, a, b, self.p)
            )
            
            return self.loss_weight * swd
        else:
            # Bidirectional case not implemented yet
            raise NotImplementedError("Bidirectional MaxSW not implemented")