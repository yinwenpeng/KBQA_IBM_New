import shlex
import sys
import re
import gzip

def refine_Q(raw_Q):
	question_mark_pos=raw_Q.find('?')
	return raw_Q[:question_mark_pos].lower() # lowercase question string

def last_slash_pos(str):
	return str.rfind('/')

def last_dot_pos(str):
	return str.rfind('.')

def refine_entity(link):
	slash_pos=link.rfind('/')
	if link[-1]==',':
		return link[slash_pos+1:-1]
	else:
		return link[slash_pos+1:]

def refine_list(answer_list):
	reg=re.compile('[\")(,]')
	candidates=answer_list.split('description')
	answers=''
	for i in range(1,len(candidates)):
		answers+='\t'+reg.sub('', candidates[i])
	return answers.strip()

def preprocess_WebQuestion(path, inputname, train):
	readfile=open(path+inputname, 'r')
	if train is True:
		writefile=open(path+'train.txt', 'w')
	else:
		writefile=open(path+'test.txt', 'w')
	count=0
	for line in readfile:
		if line.find('utterance')>=0:
			parts=shlex.split(line.strip())
			Q=refine_Q(parts[5])
			entity=refine_entity(parts[1])
			answer_list=refine_list(parts[3])
			writefile.write(Q+'\t'+entity+'\t'+answer_list+'\n')
			count+=1
	print 'totally', count, 'rows'
	writefile.close()
	readfile.close()


def	load_id2names(path,	infile):
				readfile=open(path+infile,	'r')
				id2names={}
				count=0
				for	line	in	readfile:
								parts=line.strip().split('::')
								id2names[parts[0].strip()]=parts[1].strip()
								count+=1
								#exit(0)
				print count, 'names, loaded over'
				readfile.close()
				return id2names


def convert_triples(freebase_path, path, infile, id2names):
	readfile=gzip.open(freebase_path+infile, 'r')
	count=0
	writefile=gzip.open(path+'triples.txt.gz', 'w')
	for line in readfile:
		#print line
		#exit(0)
		parts=line.strip().split()
		if parts[0].find('/m.')>=0 and parts[2].find('/m.')>=0:
			head=parts[0][last_slash_pos(parts[0].strip())+1:-1]
			relation=parts[1][last_dot_pos(parts[1].strip())+1:-1]
			tail=parts[2][last_slash_pos(parts[2].strip())+1:-1]
			#print head, relation, tail
			#if head.find('/m.')>=0 and tail.find('/m.')>=0:
			#print head, tail
			head_str=id2names.get(head)
			tail_str=id2names.get(tail)
			#print head_str, tail_str
			#exit(0)
			if head_str is not None and tail_str is not None:
				count+=1
				writefile.write(head_str+'\t'+relation+'\t'+tail_str+'\n')
				if count%10000:
					print count, '...'
				#if count==5000:
				#	exit(0)
	print 'totally', count, 'valid triples'
	writefile.close()
	readfile.close()

def convert_triples_relaxed(freebase_path, path, infile, id2names):
	readfile=gzip.open(freebase_path+infile, 'r')
	count=0
	writefile=gzip.open(path+'triples_relaxed.txt.gz', 'w')
	for line in readfile:
		#print line
		#exit(0)
		parts=line.strip().split()
		if parts[0].find('/m.')>=0 or parts[2].find('/m.')>=0:
			
			relation=parts[1][last_dot_pos(parts[1].strip())+1:-1]
			if parts[0].find('/m.')>=0:
				head=parts[0][last_slash_pos(parts[0].strip())+1:-1]
				head_str=id2names.get(head)
			else:
				head=parts[0][last_dot_pos(parts[0].strip())+1:-1]
				head_str=head
			if parts[2].find('/m.')>=0:
				tail=parts[2][last_slash_pos(parts[2].strip())+1:-1]
				tail_str=id2names.get(tail)
			else:
				tail=parts[2][last_dot_pos(parts[2].strip())+1:-1]
				tail_str=tail
			#print head, relation, tail
			#if head.find('/m.')>=0 and tail.find('/m.')>=0:
			#print head, tail
# 			head_str=id2names.get(head)
# 			tail_str=id2names.get(tail)
			#print head_str, tail_str
			#exit(0)
			if head_str is not None and tail_str is not None:
				count+=1
				writefile.write(head_str+'\t'+relation+'\t'+tail_str+'\n')
				if count%1000000==0:
					print count, '...'
				#if count==5000:
				#	exit(0)
	print 'totally', count, 'valid triples'
	writefile.close()
	readfile.close()
	
def entity_description_statistics(freebase_path, path, des, triples):
	#first load all descriptions
	des_file=open(freebase_path+des, 'r')
	names2des={}
	for line in des_file:

		parts=line.strip().split('\t')
		if len(parts)==2:
			names2des[parts[0].strip()]=parts[1].strip()
	des_file.close()
	print 'totally names2des size:', len(names2des)
	#load triples
	triple_file=gzip.open(path+triples, 'r')
	entitySet=set()
	weird_triple=0
	for line in triple_file:
# 		print line
		parts=line.strip().split('\t')
		if len(parts)==3:
			entitySet.add(parts[0].strip())
			entitySet.add(parts[2].strip())
		else:
			weird_triple+=1
	triple_file.close()
	entity_size=len(entitySet)
	print 'totally entity size:', entity_size, 'weird_triple:', weird_triple
	count=0
	for entity in entitySet:
		des=names2des.get(entity)
		if des is not None:
			count+=1
	print count, ' entities have no des', count*1.0/entity_size
			
def how_many_queryEntity_and_answerEntity_in_triples(path, triples, trainfile, testfile):  # 87.97%
	triple_file=gzip.open(path+triples, 'r')
	entitySet=set()
	weird_triple=0
	for line in triple_file:
# 		print line
		parts=line.strip().split('\t')
		if len(parts)==3:
			entitySet.add(parts[0].strip().lower())#convert into lowercase to string matching
			entitySet.add(parts[2].strip().lower())
		else:
			weird_triple+=1
	triple_file.close()
	entity_size=len(entitySet)
	print 'totally entity size:', entity_size, 'weird_triple:', weird_triple	
	files=[trainfile, testfile]
	query_answer_entitySet=set()
	for fil in files:
		line_co=0
		readfile=open(path+fil, 'r')
		for line in readfile:
			parts=line.strip().split('\t')
			queryEntity=parts[1].replace('_', ' ')
			answerEntityList=parts[2].strip().lower().split()#multiple answer entity for a question
			query_answer_entitySet.add(queryEntity)
			query_answer_entitySet=query_answer_entitySet.union(answerEntityList)
# 			line_co+=1
# 			if line_co>20:
# 				exit(0)
		readfile.close()
	intersection=entitySet&query_answer_entitySet
	in_size=len(intersection)
	all_size=len(query_answer_entitySet)
	print 'intersection size:', in_size, 'cover rato:', in_size*1.0/all_size
						
def how_many_queryEntity_and_answerEntity_in_id2names(freebase_path, path, id2names, trainfile, testfile): #89.22, cover 90 more
	triple_file=open(freebase_path+id2names, 'r')
	entitySet=set()
# 	weird_triple=0
	for line in triple_file:
# 		print line
		parts=line.strip().split('::')
		entitySet.add(parts[1].strip().lower())#convert into lowercase to string matching

	triple_file.close()
	entity_size=len(entitySet)
	print 'totally entity size:', entity_size
	files=[trainfile, testfile]
	query_answer_entitySet=set()
	for fil in files:
		line_co=0
		readfile=open(path+fil, 'r')
		for line in readfile:
			parts=line.strip().split('\t')
			queryEntity=parts[1].replace('_', ' ')
			answerEntityList=parts[2].strip().lower().split()#multiple answer entity for a question
			query_answer_entitySet.add(queryEntity)
			query_answer_entitySet=query_answer_entitySet.union(answerEntityList)
# 			line_co+=1
# 			if line_co>20:
# 				exit(0)
		readfile.close()
	intersection=entitySet&query_answer_entitySet
	in_size=len(intersection)
	all_size=len(query_answer_entitySet)
	print 'intersection size:', in_size, 'cover rato:', in_size*1.0/all_size
	
if __name__ == '__main__':
	path='/mounts/data/proj/wenpeng/Dataset/freebase/'
	freebase_path='/mounts/data/corp/freebase.com/'
	#preprocess_WebQuestion(path, 'webquestions.examples.test.json', train=False)
 	id2names=load_id2names(freebase_path, 'freebase.id2names')
#  	convert_triples(freebase_path, path, 'freebase-rdf-2014-04-13-00-00.gz', id2names)
  	convert_triples_relaxed(freebase_path, path, 'freebase-rdf-2014-04-13-00-00.gz', id2names)
# 	entity_description_statistics(freebase_path, path, 'freebase.name2description.en', 'triples.txt.gz')
# 	how_many_queryEntity_and_answerEntity_in_triples(path, 'triples.txt.gz', 'train.txt', 'test.txt')
# 	how_many_queryEntity_and_answerEntity_in_id2names(freebase_path, path, 'freebase.id2names', 'train.txt', 'test.txt')
	
