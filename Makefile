install:
	pip install --upgrade pip
	pip install -r requirements.txt
	quarto add quarto-ext/shinylive

build:
	quarto render ./ --to html 

serve:
	python -m http.server 8999 -d _site/
