# Build the Lambda paper. Run `make` to produce lambda.pdf.
#
# Prefers `tectonic` (self-contained, auto-downloads packages — install via
# `brew install tectonic`). Falls back to `pdflatex` + `bibtex` if tectonic
# isn't available (requires TeX Live / MacTeX / MiKTeX).

PAPER = lambda

.PHONY: all clean view

all: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex $(PAPER).bib
	@if command -v tectonic >/dev/null 2>&1; then \
		echo "Using tectonic..."; \
		tectonic $(PAPER).tex; \
	elif command -v pdflatex >/dev/null 2>&1; then \
		echo "Using pdflatex + bibtex..."; \
		pdflatex -interaction=nonstopmode $(PAPER).tex; \
		bibtex $(PAPER); \
		pdflatex -interaction=nonstopmode $(PAPER).tex; \
		pdflatex -interaction=nonstopmode $(PAPER).tex; \
	else \
		echo "Error: neither tectonic nor pdflatex found."; \
		echo "Install one of:"; \
		echo "  brew install tectonic       (recommended; self-contained)"; \
		echo "  brew install --cask basictex (smaller TeX Live)"; \
		echo "  brew install --cask mactex   (full TeX Live)"; \
		exit 1; \
	fi

view: $(PAPER).pdf
	open $(PAPER).pdf

clean:
	rm -f $(PAPER).aux $(PAPER).bbl $(PAPER).blg $(PAPER).log $(PAPER).out $(PAPER).pdf
