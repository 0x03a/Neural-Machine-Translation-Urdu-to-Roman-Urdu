import streamlit as st
import torch
import torch.nn as nn
import json
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import re

# Set page config
st.set_page_config(
    page_title="Urdu to Roman Urdu Translator",
    page_icon="🔤",
    layout="wide"
)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =============================================================================
# MODEL CLASSES (same as your working Kaggle code)
# =============================================================================
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(BahdanauAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.W_encoder = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_decoder = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.V = nn.Linear(hidden_dim, 1, bias=False)
    
    def forward(self, decoder_hidden, encoder_outputs, mask=None):
        encoder_proj = self.W_encoder(encoder_outputs)
        decoder_proj = self.W_decoder(decoder_hidden).unsqueeze(1)
        scores = torch.tanh(encoder_proj + decoder_proj)
        attention_scores = self.V(scores).squeeze(2)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        attention_weights = torch.softmax(attention_scores, dim=1)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attention_weights

class BiLSTMEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout_rate):
        super(BiLSTMEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout_rate)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, 
                           bidirectional=True, dropout=dropout_rate if num_layers > 1 else 0)
        self.output_projection = nn.Linear(hidden_dim * 2, hidden_dim)
        self.h_projection = nn.Linear(hidden_dim * 2, hidden_dim)
        self.c_projection = nn.Linear(hidden_dim * 2, hidden_dim)
    
    def forward(self, x, return_sequences=True):
        embedded = self.dropout(self.embedding(x))
        outputs, (hidden, cell) = self.lstm(embedded)
        encoder_outputs = self.output_projection(outputs)
        hidden = hidden.view(self.num_layers, 2, -1, self.hidden_dim)
        cell = cell.view(self.num_layers, 2, -1, self.hidden_dim)
        hidden_last = torch.cat([hidden[-1, 0], hidden[-1, 1]], dim=1)
        cell_last = torch.cat([cell[-1, 0], cell[-1, 1]], dim=1)
        hidden_projected = torch.tanh(self.h_projection(hidden_last))
        cell_projected = torch.tanh(self.c_projection(cell_last))
        return encoder_outputs, hidden_projected, cell_projected

class LSTMDecoderWithAttention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout_rate):
        super(LSTMDecoderWithAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout_rate)
        self.attention = BahdanauAttention(hidden_dim)
        self.lstm = nn.LSTM(embedding_dim + hidden_dim, hidden_dim, num_layers,
                           batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)
    
    def forward(self, x, hidden, cell, encoder_outputs, encoder_mask=None):
        batch_size = x.size(0)
        seq_len = x.size(1)
        hidden_expanded = hidden.unsqueeze(0).repeat(self.num_layers, 1, 1)
        cell_expanded = cell.unsqueeze(0).repeat(self.num_layers, 1, 1)
        embedded = self.dropout(self.embedding(x))
        outputs = []
        attention_weights_list = []
        for t in range(seq_len):
            current_embedded = embedded[:, t:t+1, :]
            context, attn_weights = self.attention(hidden_expanded[-1], encoder_outputs, encoder_mask)
            lstm_input = torch.cat([current_embedded, context.unsqueeze(1)], dim=2)
            lstm_out, (hidden_expanded, cell_expanded) = self.lstm(lstm_input, (hidden_expanded, cell_expanded))
            combined = torch.cat([lstm_out.squeeze(1), context], dim=1)
            output = self.fc(combined)
            outputs.append(output)
            attention_weights_list.append(attn_weights)
        predictions = torch.stack(outputs, dim=1)
        attention_weights = torch.stack(attention_weights_list, dim=1)
        return predictions, attention_weights

class Seq2SeqWithAttention(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2SeqWithAttention, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, encoder_input, decoder_input):
        encoder_mask = (encoder_input != 0).float()
        encoder_outputs, hidden, cell = self.encoder(encoder_input)
        outputs, attention_weights = self.decoder(decoder_input, hidden, cell, encoder_outputs, encoder_mask)
        return outputs

# =============================================================================
# LOADING FUNCTIONS (same as your working Kaggle code)
# =============================================================================
def load_vocab(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        vocab_data['idx_to_char'] = {int(k): v for k, v in vocab_data['idx_to_char'].items()}
        return vocab_data
    except Exception as e:
        st.error(f"Error loading {filepath}: {str(e)}")
        return None

def text_to_sequence(text, vocab, max_length=100):
    if vocab is None:
        return None
    char_to_idx = vocab['char_to_idx']
    sequence = [char_to_idx.get('<sos>', 2)]
    for char in text:
        sequence.append(char_to_idx.get(char, char_to_idx.get('<unk>', 1)))
    sequence.append(char_to_idx.get('<eos>', 3))
    if len(sequence) < max_length:
        sequence += [char_to_idx.get('<pad>', 0)] * (max_length - len(sequence))
    else:
        sequence = sequence[:max_length]
    return torch.tensor(sequence, dtype=torch.long)

def sequence_to_text(sequence, vocab):
    if vocab is None:
        return ""
    idx_to_char = vocab['idx_to_char']
    text = ''
    for idx in sequence:
        idx = int(idx)
        if idx in [0, 2, 3]:  # pad, sos, eos
            continue
        char = idx_to_char.get(idx, idx_to_char.get(str(idx), '<unk>'))
        if char != '<unk>':
            text += char
    return text

def is_urdu_text(text):
    """Check if text contains Urdu characters"""
    urdu_range = re.compile(
        r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]'
    )
    return bool(urdu_range.search(text))

def preprocess_input(text):
    """Clean and preprocess input text"""
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # If text appears to be Roman but should be Urdu, show warning
    if not is_urdu_text(text) and any(c.isalpha() for c in text):
        return text, "warning"
    
    return text, "ok"

def translate_simple(model, text, urdu_vocab, roman_vocab, device):
    """Simple translation function without complex preprocessing"""
    if not text.strip():
        return ""
        
    model.eval()
    with torch.no_grad():
        encoder_input = text_to_sequence(text, urdu_vocab).unsqueeze(0).to(device)
        encoder_mask = (encoder_input != 0).float()
        encoder_outputs, hidden, cell = model.encoder(encoder_input)
        
        num_layers = model.decoder.num_layers
        sos_token = roman_vocab['char_to_idx'].get('<sos>', 2)
        eos_token = roman_vocab['char_to_idx'].get('<eos>', 3)
        
        # Simple greedy decoding
        generated = [sos_token]
        hidden_exp = hidden.unsqueeze(0).repeat(num_layers, 1, 1)
        cell_exp = cell.unsqueeze(0).repeat(num_layers, 1, 1)
        
        for _ in range(100):
            decoder_input = torch.tensor([[generated[-1]]], dtype=torch.long).to(device)
            embedded = model.decoder.embedding(decoder_input)
            context, _ = model.decoder.attention(hidden_exp[-1], encoder_outputs, encoder_mask)
            lstm_input = torch.cat([embedded, context.unsqueeze(1)], dim=2)
            lstm_out, (hidden_exp, cell_exp) = model.decoder.lstm(lstm_input, (hidden_exp, cell_exp))
            combined = torch.cat([lstm_out.squeeze(1), context], dim=1)
            logits = model.decoder.fc(combined)
            next_token = torch.argmax(logits, dim=-1).item()
            
            if next_token == eos_token:
                break
            generated.append(next_token)
        
        return sequence_to_text(generated[1:], roman_vocab)

# =============================================================================
# LOAD MODEL ONCE
# =============================================================================
@st.cache_resource
def load_translator():
    """Load model and return translator function"""
    try:
        # Load vocabularies - USE THE SAME FILES AS KAGGLE
        urdu_vocab = load_vocab('Processed_Char_level_PyTorch/urdu_char_vocab.json')
        roman_vocab = load_vocab('Processed_Char_level_PyTorch/roman_char_vocab.json')
        
        if urdu_vocab is None or roman_vocab is None:
            return None, "Failed to load vocabulary files"
        
        # Load checkpoint
        try:
            checkpoint = torch.load('best_model_Exp1-BiLSTM2-LSTM4-Attention.pt', map_location=device)
        except Exception as e:
            return None, f"Error loading model file: {str(e)}"
        
        # Get vocab sizes from checkpoint
        urdu_vocab_size = checkpoint['encoder.embedding.weight'].shape[0]
        roman_vocab_size = checkpoint['decoder.embedding.weight'].shape[0]
        
        # Build model
        config = {
            'embedding_dim': 256,
            'hidden_dim': 256,
            'encoder_layers': 2,
            'decoder_layers': 4,
            'dropout': 0.2
        }

        encoder = BiLSTMEncoder(urdu_vocab_size, config['embedding_dim'], config['hidden_dim'], 
                                config['encoder_layers'], config['dropout'])
        decoder = LSTMDecoderWithAttention(roman_vocab_size, config['embedding_dim'], config['hidden_dim'],
                                       config['decoder_layers'], config['dropout'])
        model = Seq2SeqWithAttention(encoder, decoder)
        model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        
        # Return the model and vocabs for translation
        return (model, urdu_vocab, roman_vocab), "Model loaded successfully!"
        
    except Exception as e:
        return None, f"Error loading model: {str(e)}"

# =============================================================================
# STREAMLIT UI
# =============================================================================
def main():
    st.title("🔤 Urdu to Roman Urdu Translator")
    st.markdown("### Using the same code that works on Kaggle")
    st.markdown("---")
    
    # Load model
    with st.spinner("Loading model (this only happens once)..."):
        model_data, message = load_translator()
    
    if model_data is None:
        st.error(f"❌ {message}")
        st.info("Make sure you have:")
        st.info("- best_model_Exp1-BiLSTM2-LSTM4-Attention.pt")
        st.info("- Processed_Char_level_PyTorch/urdu_char_vocab.json")
        st.info("- Processed_Char_level_PyTorch/roman_char_vocab.json")
        st.info("These should be the EXACT SAME files used in your Kaggle notebook.")
        return
    
    st.success("✅ " + message)
    
    # Get model components
    model, urdu_vocab, roman_vocab = model_data
    
    # Display vocab info
    st.sidebar.subheader("Model Info")
    st.sidebar.write(f"Urdu Vocab Size: {urdu_vocab['vocab_size']}")
    st.sidebar.write(f"Roman Vocab Size: {roman_vocab['vocab_size']}")
    
    # Show test examples
    st.sidebar.subheader("Test Examples")
    test_examples = [
        "دل کے مقدمے کو ابھی جہاں جائے",
        "سلام", 
        "کیسے ہو",
        "تم کیا کر رہے ہو"
    ]
    
    for example in test_examples:
        if st.sidebar.button(f"Try: {example}"):
            st.session_state.input_text = example
    
    # Simple input section
    st.subheader("📥 Enter Urdu Text")
    
    # Text input
    urdu_text = st.text_area(
        "Type your Urdu text below:",
        height=120,
        placeholder="Example: دل کے مقدمے کو ابھی جہاں جائے",
        key="input_text",
        value="دل کے مقدمے کو ابھی جہاں جائے"  # Default good example
    )
    
    # Preprocess and check input
    if urdu_text.strip():
        processed_text, status = preprocess_input(urdu_text)
        
        if status == "warning" and not is_urdu_text(urdu_text):
            st.warning("⚠️ The input text doesn't appear to contain Urdu characters. Please make sure you're entering Urdu text for proper translation.")
            st.info("💡 Try copying and pasting one of the test examples from the sidebar.")
    
    # Translate button
    if st.button("🚀 Translate", type="primary", use_container_width=True):
        if urdu_text.strip():
            with st.spinner("Translating..."):
                try:
                    # Preprocess input
                    processed_text, status = preprocess_input(urdu_text)
                    
                    translation = translate_simple(model, processed_text, urdu_vocab, roman_vocab, device)
                    
                    # Display results
                    st.subheader("📤 Translation Result")
                    
                    if status == "warning" and not is_urdu_text(urdu_text):
                        st.error("❌ Translation may be incorrect - input doesn't appear to be Urdu text")
                    else:
                        st.success("✅ Translation complete!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Input Length", f"{len(urdu_text)} chars")
                        st.code(f"Input: {urdu_text}")
                    with col2:
                        st.metric("Output Length", f"{len(translation)} chars")
                        st.code(f"Output: {translation}")
                    
                    st.text_area(
                        "Roman Urdu Translation:",
                        value=translation,
                        height=100,
                        key="translation_output"
                    )
                    
                    # Show what a correct translation should look like
                    st.subheader("🧪 Expected Results for Common Phrases")
                    expected_results = {
                        "دل کے مقدمے کو ابھی جہاں جائے": "dil ke muqaddame ko ahn jah jase",
                        "سلام": "salaam", 
                        "کیسے ہو": "kaise ho",
                        "تم کیا کر رہے ہو": "tum kya kar rahe ho"
                    }
                    
                    for urdu, expected in expected_results.items():
                        col1, col2, col3 = st.columns([2, 2, 1])
                        with col1:
                            st.write(f"**{urdu}**")
                        with col2:
                            actual = translate_simple(model, urdu, urdu_vocab, roman_vocab, device)
                            if actual == expected:
                                st.success(f"`{actual}`")
                            else:
                                st.error(f"`{actual}` (expected: `{expected}`)")
                        with col3:
                            if actual == expected:
                                st.write("✅")
                            else:
                                st.write("❌")
                    
                except Exception as e:
                    st.error(f"Translation error: {str(e)}")
                    st.info("This might indicate a vocabulary mismatch. Make sure you're using the same vocabulary files as in Kaggle.")
        else:
            st.warning("Please enter some Urdu text to translate")
    
    # Model info
    st.markdown("---")
    st.subheader("🔧 Model Information")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Architecture", "BiLSTM + Attention")
        st.metric("Encoder Layers", "2")
    with col2:
        st.metric("Decoder Layers", "4")
        st.metric("Hidden Size", "256")
    with col3:
        st.metric("Training Data", "21,000+ samples")
        st.metric("Expected Vocab Sizes", "Urdu: 52, Roman: 31")

if __name__ == "__main__":
    main()