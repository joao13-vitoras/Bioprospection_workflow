from pathlib import Path

dir = input('Path to directory:	')

unique_embed = set()

for file in Path(dir).rglob('*.pt'):
	if file in unique_embed:
		print(f'File {file} is a duplicate')
	else:
		unique_embed.add(file)
