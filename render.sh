#!/bin/bash

TO_RENDER=(index.qmd
           intro.qmd
           explainable-ai-systems.qmd
           cha_wahcc/wahcc.qmd
           cha_odm/odm.qmd)

echo "Removing previous slides"
rm -rf slides
echo "Loading virtual environment"
source venv/bin/activate
echo "Rendering Quarto slides"
quarto render ${TO_RENDER[@]} --to revealjs 
echo "Copying slides into a separate folder"
cp -R _site slides
echo "Rendering Quarto website"
quarto render ${TO_RENDER[@]} --to html 
echo "Removing slides folder from the website"
rm -rf _site/slides
echo "Copying slides into the website"
cp -Rf slides _site/slides
echo "Removing slides original copy"
rm -rf slides
