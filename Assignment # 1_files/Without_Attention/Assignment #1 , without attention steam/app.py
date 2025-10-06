import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import json
import Levenshtein
import time
import traceback

# Set page config
st.set_page_config(
    page_title="Urdu to Roman Urdu Translator",
    page_icon="🌐",
    layout="wide"
)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =============================================================================
# MODEL ARCHITECTURE (MUST MATCH YOUR TRAINING CODE)
# =============================================================================
class BiLSTMEncoder(nn.Module):
    """BiLSTM Encoder - returns only final states"""
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout_rate):
        super(BiLSTMEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout_rate)
        
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # Project bidirectional states to single direction
        self.h_projection = nn.Linear(hidden_dim * 2, hidden_dim)
        self.c_projection = nn.Linear(hidden_dim * 2, hidden_dim)
    
    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        outputs, (hidden, cell) = self.lstm(embedded)
        
        hidden = hidden.view(self.num_layers, 2, -1, self.hidden_dim)
        cell = cell.view(self.num_layers, 2, -1, self.hidden_dim)
        
        hidden_last = torch.cat([hidden[-1, 0], hidden[-1, 1]], dim=1)
        cell_last = torch.cat([cell[-1, 0], cell[-1, 1]], dim=1)
        
        hidden_projected = torch.tanh(self.h_projection(hidden_last))
        cell_projected = torch.tanh(self.c_projection(cell_last))
        
        return hidden_projected, cell_projected

class LSTMDecoder(nn.Module):
    """Multi-layer LSTM Decoder without Attention"""
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout_rate):
        super(LSTMDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout_rate)
        
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, hidden, cell):
        hidden_expanded = hidden.unsqueeze(0).repeat(self.num_layers, 1, 1)
        cell_expanded = cell.unsqueeze(0).repeat(self.num_layers, 1, 1)
        
        embedded = self.dropout(self.embedding(x))
        output, (hidden_out, cell_out) = self.lstm(embedded, (hidden_expanded, cell_expanded))
        predictions = self.fc(output)
        
        return predictions

class Seq2Seq(nn.Module):
    """Complete Seq2Seq Model without Attention"""
    
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, encoder_input, decoder_input):
        hidden, cell = self.encoder(encoder_input)
        outputs = self.decoder(decoder_input, hidden, cell)
        return outputs

# =============================================================================
# INFERENCE CLASSES
# =============================================================================
class GreedyDecoder:
    """Simple greedy decoding"""
    
    def __init__(self, model, roman_vocab, max_length=100):
        self.model = model
        self.roman_vocab = roman_vocab
        self.max_length = max_length
        self.eos_token = roman_vocab['char_to_idx']['<eos>']
        self.sos_token = roman_vocab['char_to_idx']['<sos>']
        self.pad_token = roman_vocab['char_to_idx']['<pad>']
        self.device = next(model.parameters()).device
    
    def decode(self, encoder_input):
        self.model.eval()
        with torch.no_grad():
            encoder_input = encoder_input.unsqueeze(0).to(self.device)
            hidden, cell = self.model.encoder(encoder_input)
            
            decoder_input = torch.full((1, 1), self.sos_token, dtype=torch.long).to(self.device)
            generated_sequence = [self.sos_token]
            
            num_layers = self.model.decoder.num_layers
            hidden_expanded = hidden.unsqueeze(0).repeat(num_layers, 1, 1)
            cell_expanded = cell.unsqueeze(0).repeat(num_layers, 1, 1)
            
            for step in range(self.max_length - 1):
                embedded = self.model.decoder.dropout(
                    self.model.decoder.embedding(decoder_input)
                )
                
                lstm_out, (hidden_expanded, cell_expanded) = self.model.decoder.lstm(
                    embedded, (hidden_expanded, cell_expanded)
                )
                
                logits = self.model.decoder.fc(lstm_out[:, -1, :])
                next_token = torch.argmax(logits, dim=-1).item()
                
                if next_token == self.eos_token:
                    break
                
                generated_sequence.append(next_token)
                decoder_input = torch.tensor([[next_token]], dtype=torch.long).to(self.device)
            
            return generated_sequence[1:]

class DataPreprocessor:
    """Handles text to sequence and sequence to text conversion"""
    
    def __init__(self):
        self.urdu_vocab = None
        self.roman_vocab = None
    
    def set_vocabularies(self, urdu_vocab, roman_vocab):
        self.urdu_vocab = urdu_vocab
        self.roman_vocab = roman_vocab
    
    def text_to_sequence(self, text, vocab_type='urdu'):
        """Convert text to sequence of indices"""
        vocab = self.urdu_vocab if vocab_type == 'urdu' else self.roman_vocab
        char_to_idx = vocab['char_to_idx']
        
        sequence = []
        for char in text:
            if char in char_to_idx:
                sequence.append(char_to_idx[char])
            else:
                sequence.append(char_to_idx.get('<unk>', 1))
        
        return sequence
    
    def sequence_to_text(self, sequence, vocab_type='roman'):
        vocab = self.roman_vocab if vocab_type == 'roman' else self.urdu_vocab
        idx_to_char = vocab['idx_to_char']
        
        text = ''
        for idx in sequence:
            idx = int(idx)
            if idx in [vocab['char_to_idx']['<pad>'],
                      vocab['char_to_idx']['<sos>'],
                      vocab['char_to_idx']['<eos>']]:
                continue
            text += idx_to_char.get(idx, '<unk>')
        
        return text

# =============================================================================
# MODEL LOADING FUNCTION
# =============================================================================
@st.cache_resource
def load_model_and_vocab():
    """Load the trained model and vocabularies"""
    try:
        # Load vocabularies
        folder = "Processed_Char_level_PyTorch"
        
        with open(f"{folder}/urdu_char_vocab.json", 'r', encoding='utf-8') as f:
            urdu_vocab = json.load(f)
        with open(f"{folder}/roman_char_vocab.json", 'r', encoding='utf-8') as f:
            roman_vocab = json.load(f)
        
        # Convert idx_to_char keys to int
        urdu_vocab['idx_to_char'] = {int(k): v for k, v in urdu_vocab['idx_to_char'].items()}
        roman_vocab['idx_to_char'] = {int(k): v for k, v in roman_vocab['idx_to_char'].items()}
        
        # Model configuration (from your Exp3)
        config = {
            'embedding_dim': 256,
            'hidden_dim': 512,
            'encoder_layers': 2,
            'decoder_layers': 2,
            'dropout': 0.1
        }
        
        # Build model
        encoder = BiLSTMEncoder(
            urdu_vocab['vocab_size'],
            config['embedding_dim'],
            config['hidden_dim'],
            config['encoder_layers'],
            config['dropout']
        )
        
        decoder = LSTMDecoder(
            roman_vocab['vocab_size'],
            config['embedding_dim'],
            config['hidden_dim'],
            config['decoder_layers'],
            config['dropout']
        )
        
        model = Seq2Seq(encoder, decoder)
        
        # Load trained weights
        model_path = "best_model_Exp3-BiLSTM2-LSTM2-Large-NoAttention.pt"
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        # Create preprocessor
        preprocessor = DataPreprocessor()
        preprocessor.set_vocabularies(urdu_vocab, roman_vocab)
        
        # Create decoder
        greedy_decoder = GreedyDecoder(model, roman_vocab)
        
        return model, preprocessor, greedy_decoder, urdu_vocab, roman_vocab
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.code(traceback.format_exc())
        return None, None, None, None, None

# =============================================================================
# PHONETIC FALLBACK (SIMPLE MAPPING)
# =============================================================================
def simple_phonetic_translation(urdu_text):
    """Simple phonetic fallback translation"""
    # Basic Urdu to Roman phonetic mapping
    phonetic_map = {
        'ا': 'a', 'ب': 'b', 'پ': 'p', 'ت': 't', 'ٹ': 't', 'ث': 's',
        'ج': 'j', 'چ': 'ch', 'ح': 'h', 'خ': 'kh', 'د': 'd', 'ڈ': 'd',
        'ذ': 'z', 'ر': 'r', 'ڑ': 'r', 'ز': 'z', 'ژ': 'zh', 'س': 's',
        'ش': 'sh', 'ص': 's', 'ض': 'z', 'ط': 't', 'ظ': 'z', 'ع': 'a',
        'غ': 'gh', 'ف': 'f', 'ق': 'q', 'ک': 'k', 'گ': 'g', 'ل': 'l',
        'م': 'm', 'ن': 'n', 'و': 'w', 'ہ': 'h', 'ھ': 'h', 'ء': '',
        'ی': 'y', 'ے': 'e', ' ': ' '
    }
    
    result = ""
    for char in urdu_text:
        result += phonetic_map.get(char, char)
    return result

# =============================================================================
# TRANSLATION FUNCTION
# =============================================================================
def translate_text(urdu_text, preprocessor, decoder, use_fallback=False):
    """Translate Urdu text to Roman Urdu"""
    try:
        if use_fallback:
            return simple_phonetic_translation(urdu_text), "phonetic"
        
        # Convert text to sequence
        sequence = preprocessor.text_to_sequence(urdu_text, 'urdu')
        
        if len(sequence) == 0:
            return simple_phonetic_translation(urdu_text), "phonetic_fallback"
        
        # Convert to tensor
        encoder_input = torch.tensor(sequence, dtype=torch.long)
        
        # Decode
        predicted_seq = decoder.decode(encoder_input)
        
        # Convert back to text
        predicted_text = preprocessor.sequence_to_text(predicted_seq, 'roman')
        
        return predicted_text, "model"
        
    except Exception as e:
        st.error(f"Translation error: {str(e)}")
        return simple_phonetic_translation(urdu_text), "error_fallback"

# =============================================================================
# STREAMLIT APP
# =============================================================================
def main():
    st.title("🌐 Urdu to Roman Urdu Translator")
    st.markdown("""
    **Model Type**: Sequence-to-Sequence (BiLSTM Encoder + LSTM Decoder)  
    **Tokenization**: Character-level  
    **Model**: Exp3-BiLSTM2-LSTM2-Large-NoAttention (Best Model)  
    **Fallback**: Phonetic translation available  
    **Input Limit**: 100 characters
    """)
    
    # Load model
    with st.spinner("Loading model and vocabularies..."):
        model, preprocessor, decoder, urdu_vocab, roman_vocab = load_model_and_vocab()
    
    if model is None:
        st.error("Failed to load model. Please check if model files exist.")
        return
    
    # Display model info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"**Urdu Vocabulary**: {urdu_vocab['vocab_size']} chars")
    with col2:
        st.info(f"**Roman Vocabulary**: {roman_vocab['vocab_size']} chars")
    with col3:
        st.info(f"**Device**: {device}")
    
    st.divider()
    
    # Input section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        urdu_input = st.text_area(
            "📝 Enter Urdu Text:",
            placeholder="Type Urdu text here...",
            height=150,
            max_chars=100
        )
    
    with col2:
        st.markdown("### Options")
        show_metrics = st.checkbox("Show Translation Metrics", value=True)
    
    # Translate button
    if st.button("🚀 Translate", type="primary", use_container_width=True):
        if not urdu_input.strip():
            st.warning("Please enter some Urdu text to translate.")
        else:
            with st.spinner("Translating..."):
                start_time = time.time()
                
                # Perform translation
                translated_text, method = translate_text(
                    urdu_input.strip(), 
                    preprocessor, 
                    decoder, 
                  
                )
                
                translation_time = time.time() - start_time
            
            # Display results
            st.divider()
            st.subheader("📋 Translation Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.text_area("**Original Urdu Text:**", urdu_input, height=100)
            
            with col2:
                method_badge = {
                    "model": "🤖 Model Prediction",

                }
                
                st.text_area(
                    f"**Roman Urdu Translation** ({method_badge[method]}):", 
                    translated_text, 
                    height=100
                )
            
            # Show metrics
            if show_metrics and method == "model":
                st.divider()
                st.subheader("📊 Translation Metrics")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Input Length", f"{len(urdu_input)} chars")
                
                with col2:
                    st.metric("Output Length", f"{len(translated_text)} chars")
                
                with col3:
                    st.metric("Translation Time", f"{translation_time:.3f}s")
                
                # Calculate basic quality metrics
                if len(urdu_input) > 0 and len(translated_text) > 0:
                    cer = Levenshtein.distance(urdu_input, translated_text) / max(len(urdu_input), len(translated_text))
                    edit_distance = Levenshtein.distance(urdu_input, translated_text)
                    
                    col4, col5 = st.columns(2)
                    with col4:
                        st.metric("Character Error Rate (CER)", f"{cer:.4f}")
                    with col5:
                        st.metric("Edit Distance", f"{edit_distance}")
    
    # Example section
    st.divider()
 
  
    
    # Technical details (collapsible)
    with st.expander("🔧 Technical Details"):
        st.markdown("""
        **Model Architecture:**
        - **Encoder**: 2-layer BiLSTM (512 hidden dim)
        - **Decoder**: 2-layer LSTM (512 hidden dim) 
        - **Embedding**: 256 dimensions
        - **Vocab Size**: Urdu - 20+, Roman - 20+ characters
        - **Training**: Character-level tokenization, No Attention mechanism
        
        **Training Configuration (Exp3):**
        - Learning Rate: 0.0005
        - Batch Size: 32
        - Dropout: 0.1
        - Epochs: 20 (with early stopping)
        
        **Inference:**
        - Greedy decoding (no beam search)
        - Max sequence length: 100 characters
        - Automatic phonetic fallback
        """)
        
        if st.button("Clear Cache", type="secondary"):
            st.cache_resource.clear()

if __name__ == "__main__":
    main()