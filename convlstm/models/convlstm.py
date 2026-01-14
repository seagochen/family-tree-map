import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """
    Single ConvLSTM cell that processes one timestep.
    
    Implements the equations:
        i_t = sigmoid(W_xi * x_t + W_hi * h_{t-1} + b_i)
        f_t = sigmoid(W_xf * x_t + W_hf * h_{t-1} + b_f)
        o_t = sigmoid(W_xo * x_t + W_ho * h_{t-1} + b_o)
        g_t = tanh(W_xg * x_t + W_hg * h_{t-1} + b_g)
        c_t = f_t * c_{t-1} + i_t * g_t
        h_t = o_t * tanh(c_t)
    
    Where * denotes convolution operation.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int, bias: bool = True):
        """
        Args:
            input_dim: Number of channels in input tensor
            hidden_dim: Number of channels in hidden state
            kernel_size: Size of convolutional kernel
            bias: Whether to add bias
        """
        super(ConvLSTMCell, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias
        
        # Combined convolution for input (computes all 4 gates at once)
        self.conv_x = nn.Conv2d(
            in_channels=input_dim,
            out_channels=4 * hidden_dim,  # i, f, o, g gates
            kernel_size=kernel_size,
            padding=self.padding,
            bias=bias
        )
        
        # Combined convolution for hidden state
        self.conv_h = nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=False  # Bias already in conv_x
        )
    
    def forward(self, x, hidden_state):
        """
        Args:
            x: Input tensor of shape (batch, input_dim, height, width)
            hidden_state: Tuple of (h, c) each of shape (batch, hidden_dim, height, width)
        
        Returns:
            h_next, c_next: Next hidden and cell states
        """
        h_cur, c_cur = hidden_state
        
        # Compute gates
        combined_x = self.conv_x(x)
        combined_h = self.conv_h(h_cur)
        combined = combined_x + combined_h
        
        # Split into 4 gates
        cc_i, cc_f, cc_o, cc_g = torch.split(combined, self.hidden_dim, dim=1)
        
        i = torch.sigmoid(cc_i)  # Input gate
        f = torch.sigmoid(cc_f)  # Forget gate
        o = torch.sigmoid(cc_o)  # Output gate
        g = torch.tanh(cc_g)     # Cell gate
        
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next
    
    def init_hidden(self, batch_size, height, width, device):
        """Initialize hidden state with zeros."""
        return (
            torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
            torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        )


class ConvLSTM(nn.Module):
    """
    ConvLSTM module that processes a sequence of feature maps.
    
    Can be configured with multiple layers for deeper temporal modeling.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        kernel_size: int = 3,
        num_layers: int = 1,
        batch_first: bool = True,
        bias: bool = True,
        return_all_layers: bool = False
    ):
        """
        Args:
            input_dim: Number of channels in input
            hidden_dim: Number of channels in hidden state (can be int or list for multi-layer)
            kernel_size: Size of conv kernel (can be int or list)
            num_layers: Number of stacked ConvLSTM layers
            batch_first: If True, input shape is (batch, time, channels, height, width)
            bias: Whether to use bias in convolutions
            return_all_layers: If True, return outputs from all layers
        """
        super(ConvLSTM, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim if isinstance(hidden_dim, list) else [hidden_dim] * num_layers
        self.kernel_size = kernel_size if isinstance(kernel_size, list) else [kernel_size] * num_layers
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.return_all_layers = return_all_layers
        
        # Build ConvLSTM cells for each layer
        cell_list = []
        for i in range(num_layers):
            cur_input_dim = input_dim if i == 0 else self.hidden_dim[i - 1]
            cell_list.append(
                ConvLSTMCell(
                    input_dim=cur_input_dim,
                    hidden_dim=self.hidden_dim[i],
                    kernel_size=self.kernel_size[i],
                    bias=bias
                )
            )
        self.cell_list = nn.ModuleList(cell_list)
    
    def forward(self, x, hidden_state=None):
        """
        Args:
            x: Input tensor of shape (batch, time, channels, height, width) if batch_first
               or (time, batch, channels, height, width) otherwise
            hidden_state: List of tuples [(h_0, c_0), ...] for each layer, or None
        
        Returns:
            layer_output: Output tensor of shape (batch, time, hidden_dim, height, width)
            last_state_list: List of (h_n, c_n) tuples for each layer
        """
        if not self.batch_first:
            x = x.permute(1, 0, 2, 3, 4)  # (time, batch, ...) -> (batch, time, ...)
        
        batch_size, seq_len, _, height, width = x.size()
        device = x.device
        
        # Initialize hidden states if not provided
        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size, height, width, device)
        
        layer_output_list = []
        last_state_list = []
        
        cur_layer_input = x
        
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            
            # Process each timestep
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](cur_layer_input[:, t, :, :, :], (h, c))
                output_inner.append(h)
            
            # Stack outputs: (batch, time, hidden_dim, height, width)
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output
            
            layer_output_list.append(layer_output)
            last_state_list.append((h, c))
        
        if self.return_all_layers:
            return layer_output_list, last_state_list
        else:
            return layer_output_list[-1], last_state_list[-1]
    
    def _init_hidden(self, batch_size, height, width, device):
        """Initialize hidden states for all layers."""
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, height, width, device))
        return init_states