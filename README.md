# Numerical Solution of PDEs

## Building LaTeX source

To continuously build and view the TeX source:

    latexmk -lualatex -pvc -output-directory=build/ main.tex

## Converting to Markdown

To convert the tex file to Markdown (WIP):

    pandoc -s -C -t markdown-citations --from latex --to markdown --bibliography=ref.bib main.tex 
