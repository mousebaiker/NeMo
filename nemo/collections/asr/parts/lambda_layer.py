# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
import torch.nn as nn


class LambdaConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, in_context_channels, key_size, rel_conv_kernel_size):
        super().__init__()

        self.in_channels = in_channels
        self.key_size = key_size

        self.query_proj = nn.Conv1d(in_channels, key_size, kernel_size=1)
        self.key_proj = nn.Conv1d(in_context_channels, key_size, kernel_size=1)
        self.val_proj = nn.Conv1d(in_context_channels, out_channels, kernel_size=1)

        self.key_softmax = nn.Softmax(dim=-1)
        self.query_bn  = nn.BatchNorm1d(key_size)
        self.value_bn  = nn.BatchNorm1d(out_channels)

        self.positional_conv = nn.Conv2d(1, key_size, (rel_conv_kernel_size, 1), padding=(rel_conv_kernel_size//2, 0))
    
    
    def init_weights(self):
        nn.init.normal_(self.query_proj.weight, 0.0, self.key_size * self.in_channels**(-0.5))
        nn.init.normal_(self.key_proj.weight, 0.0, self.in_channels**(-0.5))
        nn.init.normal_(self.val_proj.weight, 0.0, self.in_channels**(-0.5))
        nn.init.normal_(self.positional_conv.weight)


    def forward(self, x, context):
        """
        Arguments:
          x: Tensor of shape (B, in_channels, T)
          context: Tensor of shape (B, in_context_channels, L(=T))
        
        Output:
          Tensor of shape (B, out_channels, T)
        """
        queries = self.query_proj(x) # B, K, T
        keys = self.key_proj(context) # B, K, L
        values = self.val_proj(context) # B, V, L

        norm_keys = self.key_softmax(keys)
        queries = self.query_bn(queries)
        values = self.value_bn(values)

        queries = torch.transpose(queries, -2, -1) # B, T, K
        values = torch.transpose(values, -2, -1) # B, L, V

        content_lambda = torch.bmm(norm_keys, values) # B, K, V
        positional_lambda = self.positional_conv(torch.unsqueeze(values, 1)) # B, K, L, V
        positional_lambda = torch.transpose(positional_lambda, 1, 2) # B, L, K, V

        content_output = torch.bmm(queries, content_lambda) # B, T, V
        positional_output = torch.matmul(torch.unsqueeze(queries, 2), positional_lambda)  # B, T, 1, V
        positional_output = torch.squeeze(positional_output, dim=2) # B, T, V

        output = torch.transpose(content_output + positional_output, -2, -1) # B, V, T
        return output

        
class LambdaConvSelfAttention(LambdaConvLayer):
  def __init__(self, in_channels, out_channels, key_size, rel_conv_kernel_size):
      super().__init__(in_channels, out_channels, in_channels, key_size, rel_conv_kernel_size)
  
  
  def forward(self, x):
    return super().forward(x, x)