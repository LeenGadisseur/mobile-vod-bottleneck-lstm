

fseq = open("test_EPFL_list.txt", "r")
fseq_out = open("test_EPFL_list_correct.txt", "w")

for line in fseq:
	new_line = line.replace('.JPEG', '')
	fseq_out.write(new_line)

fseq.close()
fseq_out.close()

fseq_out = open("test_EPFL_list_correct.txt", "r")
print(fseq_out.read())
