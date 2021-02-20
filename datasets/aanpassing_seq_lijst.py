

f = open("test_VID_seqs_list.txt", "r")
fout = open("test_VID_seqs_list_correct.txt", "w")

for line in f:
	new_line = line[:24]+"/"+line[24:]
	print(new_line)
	fout.write(new_line)

f.close()
fout.close()

fout = open("test_VID_seqs_list_correct.txt", "r")
print(fout.read())

