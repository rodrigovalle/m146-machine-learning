PS_TARGET=ps2_problems.tex
PS_CODE_TARGET=ps2_code.md
NAME=$(basename $(TARGET))
EXT=log aux pdf

default: ps_code

ps:
	pdflatex $(PS_TARGET)

ps_code:
	pandoc $(PS_CODE_TARGET) -o $(PS_CODE_TARGET:.md=.pdf)

clean:
	for i in $(EXT); do find . -name "$(NAME).$$i" -delete; done

.PHONY: clean
