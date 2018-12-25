TARGET=ps1.tex
NAME=$(basename $(TARGET))
EXT=log aux pdf

all:
	pdflatex $(TARGET)

clean:
	for i in $(EXT); do find . -name "$(NAME).$$i" -delete; done

.PHONY: clean
