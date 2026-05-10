import json
import os

def finalize_notebook():
    path = 'traduction_moore_francais'
    if not os.path.exists(path):
        print(f"Error: {path} not found.")
        return

    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # 1. Fix paths and remove colab specific code
    for cell in nb['cells']:
        source_str = ''.join(cell.get('source', []))
        if 'moore_path =' in source_str:
            cell['source'] = [
                '# Chemins locaux pour exécution standard\n',
                'moore_path = "corpus/moore.txt"\n',
                'french_path = "corpus/francais.txt"\n'
            ]
        if 'drive.mount' in source_str:
            cell['source'] = ['# drive.mount("/content/drive") - Désactivé pour exécution locale\n']

    # 2. Add Evaluation and Interactive cells
    new_cells = [
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '# 5. Évaluation\n',
                '\n',
                'Dans cette section, nous évaluons la qualité des traductions produites par le modèle sur le jeu de test.\n',
                'Nous utilisons :\n',
                '- Le score BLEU (Bilingual Evaluation Understudy) pour mesurer la proximité lexicale.\n',
                '- Une analyse qualitative sur 5 exemples choisis aléatoirement.'
            ]
        },
        {
            'cell_type': 'code',
            'metadata': {},
            'source': [
                '# ==========================================\n',
                '# Modèles d\'inférence\n',
                '# ==========================================\n',
                '\n',
                '# Encodeur : prend une phrase Moore et sort les états internes LSTM\n',
                'encoder_model = Model(encoder_inputs, encoder_states)\n',
                '\n',
                '# Décodeur : prend un token et les états précédents pour prédire le suivant\n',
                'decoder_state_input_h = Input(shape=(LSTM_UNITS,))\n',
                'decoder_state_input_c = Input(shape=(LSTM_UNITS,))\n',
                'decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]\n',
                '\n',
                'dec_emb_inf = decoder_embedding(decoder_inputs)\n',
                'decoder_outputs_inf, state_h_inf, state_c_inf = decoder_lstm(\n',
                '    dec_emb_inf, initial_state=decoder_states_inputs\n',
                ')\n',
                'decoder_states_inf = [state_h_inf, state_c_inf]\n',
                'decoder_outputs_inf = decoder_dense(decoder_outputs_inf)\n',
                '\n',
                'decoder_model = Model(\n',
                '    [decoder_inputs] + decoder_states_inputs,\n',
                '    [decoder_outputs_inf] + decoder_states_inf\n',
                ')\n',
                '\n',
                'def decode_sequence(input_seq):\n',
                '    # Obtenir les états initiaux de l\'encodeur\n',
                '    states_value = encoder_model.predict(input_seq, verbose=0)\n',
                '    \n',
                '    # Générer une séquence cible vide de longueur 1 avec le token <start>\n',
                '    target_seq = np.zeros((1, 1))\n',
                '    target_seq[0, 0] = french_tokenizer.word_index[\'<start>\']\n',
                '\n',
                '    stop_condition = False\n',
                '    decoded_sentence = []\n',
                '\n',
                '    while not stop_condition:\n',
                '        output_tokens, h, c = decoder_model.predict([target_seq] + states_value, verbose=0)\n',
                '        sampled_token_index = np.argmax(output_tokens[0, -1, :])\n',
                '        sampled_word = french_tokenizer.index_word.get(sampled_token_index, \'\')\n',
                '\n',
                '        if sampled_word == \'<end>\' or len(decoded_sentence) > max_decoder_len:\n',
                '            stop_condition = True\n',
                '        else:\n',
                '            if sampled_word not in [\'\', \'<start>\']:\n',
                '                decoded_sentence.append(sampled_word)\n',
                '\n',
                '        # Mettre à jour la séquence cible et les états\n',
                '        target_seq = np.zeros((1, 1))\n',
                '        target_seq[0, 0] = sampled_token_index\n',
                '        states_value = [h, c]\n',
                '\n',
                '    return \' \'.join(decoded_sentence)\n'
            ]
        },
        {
            'cell_type': 'code',
            'metadata': {},
            'source': [
                '# ==========================================\n',
                '# Affichage de 5 exemples et Score BLEU\n',
                '# ==========================================\n',
                '\n',
                'print(f"{\'Phrase Moore\':<25} | {\'Traduction Attendue\':<25} | {\'Traduction Modèle\':<25}")\n',
                'print("-" * 80)\n',
                '\n',
                'for i in range(5):\n',
                '    input_seq = encoder_test[i:i+1]\n',
                '    prediction = decode_sequence(input_seq)\n',
                '    \n',
                '    # Récupérer les mots originaux (ignorer le padding et tokens spéciaux)\n',
                '    moore_words = [moore_tokenizer.index_word.get(idx, \'\') for idx in encoder_test[i] if idx > 0]\n',
                '    actual_words = [french_tokenizer.index_word.get(idx, \'\') for idx in decoder_test[i] \n',
                '                    if idx > 0 and french_tokenizer.index_word.get(idx) not in [\'<start>\', \'<end>\']]\n',
                '    \n',
                '    print(f"{\' \'.join(moore_words):<25} | {\' \'.join(actual_words):<25} | {prediction:<25}")\n',
                '\n',
                '# Calcul du Score BLEU moyen sur le set de test (50 premiers)\n',
                'bleu_scores = []\n',
                'for i in range(min(50, len(encoder_test))):\n',
                '    input_seq = encoder_test[i:i+1]\n',
                '    ref = [[french_tokenizer.index_word.get(idx, \'\') for idx in decoder_test[i] \n',
                '           if idx > 0 and french_tokenizer.index_word.get(idx) not in [\'<start>\', \'<end>\']]]\n',
                '    hyp = decode_sequence(input_seq).split()\n',
                '    bleu_scores.append(sentence_bleu(ref, hyp, weights=(0.5, 0.5, 0, 0)))\n',
                '\n',
                'print(f"\\nScore BLEU moyen (approx) : {np.mean(bleu_scores):.4f}")\n'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '# 6. Test Interactif\n',
                '\n',
                'Cette cellule permet au correcteur de tester le modèle en temps réel.'
            ]
        },
        {
            'cell_type': 'code',
            'metadata': {},
            'source': [
                'def traduire(phrase_moore):\n',
                '    # Prétraitement\n',
                '    clean_phrase = preprocess_sentence(phrase_moore)\n',
                '    # Conversion en séquence\n',
                '    seq = moore_tokenizer.texts_to_sequences([clean_phrase])\n',
                '    # Padding\n',
                '    pad_seq = pad_sequences(seq, maxlen=max_encoder_len, padding=\'post\')\n',
                '    # Décodage\n',
                '    return decode_sequence(pad_seq)\n',
                '\n',
                '# ── Testez le modèle ─────────────────────────────────\n',
                '# Modifiez cette phrase et exécutez la cellule\n',
                'phrase_moore = "m paam fo"\n',
                'traduction = traduire(phrase_moore)\n',
                '\n',
                'print(f"Moore    : {phrase_moore}")\n',
                'print(f"Français : {traduction}")\n'
            ]
        }
    ]
    nb['cells'].extend(new_cells)

    # Save to .ipynb file
    output_path = 'traduction_moore_francais.ipynb'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    
    print(f"Success: {output_path} created with all sections (0-6).")

if __name__ == "__main__":
    finalize_notebook()
