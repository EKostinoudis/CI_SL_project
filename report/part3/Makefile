build_data_files = report.aux report.log

build: report.tex
	xelatex $? && xelatex $?

clean:
	rm ${build_data_files}

