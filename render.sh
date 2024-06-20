#!/bin/bash

TO_RENDER=(index.qmd
           intro.qmd
           explainable-ai-systems.qmd
           cha_wahcc/wahcc.qmd
           cha_odm/odm.qmd)

quarto render ${TO_RENDER[@]} 


