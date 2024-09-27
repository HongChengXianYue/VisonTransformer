import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.grid_size = img_size // patch_size
        self.projection = nn.Linear(patch_size * patch_size * in_channels, embed_dim)

    def forward(self, x):
        # 将输入图像划分为patches
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(x.size(0), -1, self.patch_size * self.patch_size * x.size(1))
        return self.projection(patches)


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * -(torch.log(torch.tensor(10000.0)) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :]


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (
                self.head_dim * num_heads == embed_dim
        ), "Embedding dimension must be divisible by number of heads"

        self.embed_dim = embed_dim  # 添加这一行

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.fc_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        N, seq_length, _ = x.size()
        queries = self.query(x).view(N, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        keys = self.key(x).view(N, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        values = self.value(x).view(N, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        energy = torch.einsum("nhqd,nhkd->nhqk", [queries, keys])
        attention = F.softmax(energy / (self.head_dim ** 0.5), dim=3)

        out = torch.einsum("nhql,nhld->nhqd", [attention, values]).reshape(N, seq_length, self.embed_dim)
        return self.fc_out(out)


class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.feed_forward = FeedForward(embed_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attention = self.attention(x)
        x = self.norm1(x + attention)  # Residual connection
        forward = self.feed_forward(x)
        return self.norm2(x + forward)  # Residual connection


class VisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, num_classes, embed_dim, num_heads, hidden_dim, num_layers):
        super(VisionTransformer, self).__init__()
        self.patch_embedding = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, hidden_dim) for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.positional_encoding(x)

        for block in self.transformer_blocks:
            x = block(x)

        x = x.mean(dim=1)  # 使用平均池化进行分类
        return self.classifier(x)


# 示例用法
if __name__ == "__main__":
    model = VisionTransformer(img_size=224, patch_size=16, in_channels=3,
                              num_classes=10, embed_dim=768,
                              num_heads=12, hidden_dim=3072, num_layers=12)
    x = torch.randn(1, 3, 224, 224)  # 示例输入
    output = model(x)
    print(output.shape)  # 应该是 [1, 10]
