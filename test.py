#!/usr/bin/env python
# _*_ coding:utf-8 _*_

def get_chunks_cws(seq):
    chunks = []
    labels = ['B', 'M', 'E', 'S']
    chunk_type, chunk_start = None, None
    for i, tag in enumerate(seq):
        if tag == labels.index('B'):
            chunk_start = i
            chunk_type = "Mu"
        elif tag == labels.index('E'):
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
        elif tag == labels.index('S'):
            chunk_type = "Si"
            chunk = (chunk_type, i, i)
            chunks.append(chunk)
        else:
            pass
    return chunks 

ass1 = get_chunks_cws([0,2,0,1,2,3,3,0,1,2])
ass2 = get_chunks_cws([0,3,1,1,2,3,3,0,1,2])
ass3 = get_chunks_cws([0,1,2,0,2,3,3,0,1,3])
ass4 = get_chunks_cws([0,1,1,1,1,1,1,1,1,1])
ass5 = get_chunks_cws([0,0,0,0,0,0,0,0,0,0])
ass6 = get_chunks_cws([1,1,1,1,1,1,1,1,1,1])
ass7 = get_chunks_cws([1,1,1,1,2,3,3,0,1,1])
print(ass1)
print(ass2)
print(ass3)
print(ass4)
print(ass5)
print(ass6)
print(ass7)
