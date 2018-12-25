SRC=ps3.md
TARGET=$(SRC:.md=.pdf)

all:
	pandoc -f markdown+tex_math_single_backslash $(SRC) -o $(TARGET)
