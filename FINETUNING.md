Dataset avec des questions réponses courtes en français
https://huggingface.co/datasets/FreedomIntelligence/alpaca-gpt4-french

Autre dataset, semble plus orienté science et programmation
https://huggingface.co/datasets/FreedomIntelligence/evol-instruct-french/

Multi-turn
https://huggingface.co/datasets/FreedomIntelligence/sharegpt-french

Contexte long :
https://huggingface.co/datasets/R0k1e/UltraLink
https://github.com/OpenBMB/UltraLink?tab=readme-ov-file


https://huggingface.co/datasets/angeluriot/french_instruct

# Code
Exemple 
https://github.com/liyuan24/deepseek_from_scratch/blob/main/datacollator.py
https://github.com/rasbt/LLMs-from-scratch/blob/main/ch07/01_main-chapter-code/gpt_instruction_finetuning.py
https://qwen.readthedocs.io/en/v1.5/training/SFT/example.html

# algorithme

## Chat template
Récupérer des données au format
{"role": "user", "content": "What color is the sky?"},
{"role": "assistant", "content": "It is blue."}

créer un template utilisant les tokens <user> et <assistant>
<user>What color is the sky?<assistant>It is blue.

## Entrainement
Durant l’entrainement, il ne faut pas faire d’apprentissage sur la question. 
<user>What color is the sky?<assistant> -> It
<user>What color is the sky?<assistant>It -> is
<user>What color is the sky?<assistant>It is -> blue
Only compute loss on the assistant's response tokens

Load conversation datasets (like ShareGPT, OpenAssistant, etc.)
Convert conversations to your format: <user>question<bot>answer
Create attention masks and loss masks
Loss mask: 1 for assistant tokens, 0 for user tokens
Handle multi-turn conversations by alternating user/bot tokens

## Utilisation
Puis quand on veut utiliser le modèle en mode Instruct on construit:
<user>Question<assistant>


I have implemented a custom LLM for educational purposes. The pre-training is finished, i will go to supervised fine-tuning.
in my custom tokenizer i have created special tokens for diaglogs <user> and <bot> if needed i also have spare special tokens.

My understanding is that :

- I need to download question / answer pairs from a dataset
- extract question and answer, tokenize text and add the special tokens to mark the begining of the question and of the answer.
{"role": "user", "content": "What color is the sky?"},
{"role": "assistant", "content": "It is blue."}
Becomes
<user>What color is the sky?<bot>It is blue.
- train with this data
I have several questions on the SFT process
- should i pad my inputs to my context length (1024) ?
- do i need to compute the index of the <bot> token in order to only input sequences starting from this token. So that the network doesn’t learn the question.
<user>What color is the sky?<bot> -> It
<user>What color is the sky?<bot>It -> is
<user>What color is the sky?<bot>It is -> blue
- explain the SFT steps required to implement it from scratch in pytorch. Do not provide the code, just explain the main steps and key ideas.

## Hugging face source code
https://github.com/huggingface/trl/blob/main/trl/trainer/sft_trainer.py
