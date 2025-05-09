# Datasets
https://www.kaggle.com/datasets/wikimedia-foundation/wikipedia-structured-contents
Download frwiki_namespace_0_0.jsonl
Le texte des articles se trouve dans dans un champ `value` qui est de ```"type": "paragraph"```
Exemple
```
"type": "paragraph", "value": "Esochí (grec moderne : Εσοχή) est une localité située dans le dème d'Arrianá, dans le district régional de Rhodope, dans la periphérie de Macédoine-Orientale-et-Thrace, en Grèce."
```
Les paragraphes sont dans des sections.
```
"sections": [{"type": "section", "name": "Abstract", "has_parts": ["type": "paragraph", "value": "Esochí etc.. ", "links": [{"url": ""}]]}]
```

# install
uv init