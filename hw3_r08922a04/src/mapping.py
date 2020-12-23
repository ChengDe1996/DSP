import sys

table = {}

input_file = sys.argv[1]
output_file = sys.argv[2]
# input_file = '../Big5-ZhuYin.map'
# output_file = '../res/ZhuYin-Big5.map'

with open(input_file, 'r', encoding = 'BIG5-HKSCS') as file:
	lines = file.readlines()
	for line in lines:
		big5 = line.split()[0]
		zhuyins = line.split()[1].split('/')

		for zhuyin in zhuyins:
			if zhuyin[0] not in table:
				table[zhuyin[0]] = set()
			table[zhuyin[0]].add(big5)
		table[big5] = big5

with open(output_file, 'w', encoding = 'BIG5-HKSCS') as file:
	for zhuyin, big5 in table.items():
		file.write(zhuyin + ' ')
		file.write(' '.join(list(big5)))
		file.write('\n')
