build_data_files = presentation.bbl presentation.blg presentation.nav presentation.snm presentation.toc presentation.aux presentation.bcf presentation.log presentation.out presentation.run.xml

build: presentation.tex
	xelatex $? && biber presentation && xelatex $?

clean:
	rm ${build_data_files}

