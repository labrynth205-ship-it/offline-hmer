import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class EncoderCNN(nn.Module):
    """
    CNN-based encoder for extracting visual features from handwritten math expressions
    """

    def __init__(self, enc_hidden_size=256):
        super(EncoderCNN, self).__init__()

        resnet = models.resnet18(weights=None)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        self.conv_reduce = nn.Conv2d(512, enc_hidden_size, kernel_size=1)

    def forward(self, images):
        """
        Extract features from input images; then reshape for attention mechanism

        images: [batch_size, channels, height, width]
        """
        features = self.resnet(images)

        features = self.conv_reduce(features)

        batch_size = features.size(0)
        feature_size = features.size(1)
        height, width = features.size(2), features.size(3)

        features = features.permute(0, 2, 3, 1).contiguous()
        features = features.view(batch_size, height*width, feature_size)

        return features


class BahdanauAttention(nn.Module):
    """
    Bahdanau attention mechanism with coverage

    encoder_att: projects encoder output to the attention dimension

    decoder_att: projects decoder hidden state to the attention dimension

    coverage_att: projects the coverage vector to the attention dimension

    full_att: computes a scalar attention score from the combined feature
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(BahdanauAttention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.coverage_att = nn.Linear(1, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)

    def forward(self, encoder_out, decoder_hidden, coverage=None):
        """
        Calculate context vector for the current time step

        encoder_out: [batch_size, num_pixels, encoder_dim]

        decoder_hidden: [batch_size, decoder_dim]

        coverage: [batch_size, num_pixels, 1]
        """

        num_pixels = encoder_out.size(1)


        encoder_att = self.encoder_att(encoder_out)

        decoder_att = self.decoder_att(decoder_hidden)
        decoder_att = decoder_att.unsqueeze(1)

        if coverage is None:
            coverage = torch.zeros(encoder_out.size(0), num_pixels, 1).to(encoder_out.device)
        coverage_att = self.coverage_att(coverage)


        att = torch.tanh(encoder_att + decoder_att + coverage_att)

        att = self.full_att(att).squeeze(2)
        alpha = F.softmax(att, dim=1)

        context_vector = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)

        coverage = coverage + alpha.unsqueeze(2)

        return context_vector, alpha, coverage


class DecoderRNN(nn.Module):
    """
    LSTM-based decoder with attention
    """

    def __init__(self, vocab_size, embed_size, encoder_dim, decoder_dim, attention_dim, dropout=0.5):
        super(DecoderRNN, self).__init__()

        self.vocab_size = vocab_size
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.attention_dim = attention_dim

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = BahdanauAttention(
            encoder_dim, decoder_dim, attention_dim)

        self.lstm_cell = nn.LSTMCell(embed_size + encoder_dim, decoder_dim)

        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)

        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.fc = nn.Linear(decoder_dim, vocab_size)
        self.dropout = nn.Dropout(p=dropout)

    def init_hidden_state(self, encoder_out):
        """
        Initialize hidden state and cell state for the LSTM

        encoder_out: [batch_size, num_pixels, encoder_dim]
        """

        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)

        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        Forward pass for training

        encoder_out: [batch_size, num_pixels, encoder_dim]

        encoded_captions: [batch_size, max_caption_length]

        caption_lengths: [batch_size, 1]
        """
        batch_size = encoder_out.size(0)
        num_pixels = encoder_out.size(1)

        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        embeddings = self.embedding(encoded_captions)

        h, c = self.init_hidden_state(encoder_out)

        decode_lengths = (caption_lengths - 1).tolist()

        predictions = torch.zeros(batch_size, max(decode_lengths), self.vocab_size).to(encoder_out.device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(encoder_out.device)

        coverage = torch.zeros(batch_size, num_pixels, 1).to(encoder_out.device)
        coverage_seq = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(encoder_out.device)

        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])

            context_vector, alpha, coverage = self.attention(
                encoder_out[:batch_size_t],
                h[:batch_size_t],
                coverage[:batch_size_t] if t > 0 else None
            )

            coverage_seq[:batch_size_t, t, :] = coverage.squeeze(2)

            gate = torch.sigmoid(self.f_beta(h[:batch_size_t]))
            context_vector = gate * context_vector

            h, c = self.lstm_cell(
                torch.cat([embeddings[:batch_size_t, t, :], context_vector], dim=1),
                (h[:batch_size_t], c[:batch_size_t])
            )

            preds = self.fc(self.dropout(h))
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, alphas, coverage_seq, decode_lengths, sort_ind

    def generate_caption(self, encoder_out, max_length=150, start_token=1, end_token=2):
        """
        Generate captions (LaTeX sequences)

        note: prediction will not have start_token

        encoder_out: [1, num_pixels, encoder_dim]
        """
        batch_size = encoder_out.size(0)
        assert batch_size == 1, "batch prediction is not supported"

        predictions = []
        alphas = []

        h, c = self.init_hidden_state(encoder_out)

        prev_word = torch.LongTensor([start_token]).to(encoder_out.device)

        coverage = None

        for i in range(max_length):
            embeddings = self.embedding(prev_word)

            context_vector, alpha, coverage = self.attention(
                encoder_out,
                h,
                coverage
            )

            gate = torch.sigmoid(self.f_beta(h))
            context_vector = gate * context_vector

            h, c = self.lstm_cell(
                torch.cat([embeddings, context_vector], dim=1),
                (h, c)
            )

            preds = self.fc(h)

            _, next_word = torch.max(preds, dim=1)

            predictions.append(next_word.item())
            alphas.append(alpha)

            prev_word = next_word

            if next_word.item() == end_token:
                break

        return predictions, alphas


class WAP(nn.Module):
    """
    Watch, Attend and Parse model for handwritten mathematical expression recognition
    """

    def __init__(self, vocab_size, embed_size=256, encoder_dim=256, decoder_dim=512, attention_dim=256, dropout=0.5):
        super(WAP, self).__init__()

        self.encoder = EncoderCNN(enc_hidden_size=encoder_dim)
        self.decoder = DecoderRNN(
            vocab_size=vocab_size,
            embed_size=embed_size,
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim,
            attention_dim=attention_dim,
            dropout=dropout
        )

    def forward(self, images, encoded_captions, caption_lengths):
        """
        Forward pass

        images: [batch_size, channels, height, width]

        encoded_captions: [batch_size, max_caption_length]

        caption_lengths: [batch_size, 1]
        """
        encoder_out = self.encoder(images)

        predictions, alphas, coverage_seq, decode_lengths, sort_ind = self.decoder(
            encoder_out, encoded_captions, caption_lengths
        )

        return predictions, alphas, coverage_seq, decode_lengths, sort_ind

    def recognize(self, image, max_length=150, start_token=1, end_token=2):
        """
        Recognize handwritten mathematical expression and output LaTeX sequence

        image: [1, channels, height, width]
        """
        batch_size = image.size(0)
        assert batch_size == 1, "batch prediction is not supported"
        encoder_out = self.encoder(image)

        predictions, alphas = self.decoder.generate_caption(
            encoder_out,
            max_length=max_length,
            start_token=start_token,
            end_token=end_token
        )

        return predictions, alphas






def main():
    my_model = EncoderCNN(256)
    example_image = torch.randn(1, 3, 224, 224)
    output = my_model.forward(example_image)
    print(output)
    print(output.shape)


if __name__ == '__main__':
    main()
