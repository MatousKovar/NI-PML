## Data 
- paper uses Amazon review dataset and amazon meta dataset

## Main topic of paper 
- Collaborative Alignment for Recommendation
- Collaborative - group similar users-item interactions together
- you take the user-item interactions and combine it with knowledge from item embeddings
- The user-item interaction and embeddings of item live in completely separate spaces and the target is to put both of these into one space - hence the alignment in the title
- second goal is not to distort the semantic meaning of item embedding  - in space similar items stay similar even when adding the user-interaction knowledge, I do not want to scramble the whole space - that is important for cold start - when there is no user interaction the item alone has to be valuable enough to recommend

## Representing data 
- from matrix it is reformatted into bipartite graph - users on left, items on right, edge means interaction
- items will also have semantic embedding 
- so there are 2 inputs - item-user interactions and item embeddings

## Goal
- new representation-learning method for improved reccomendation

## How are items represented
- title, brand and cathegory concatenated passed into llm to generate embedding 
- the initial embeddings are frozen - during training they are not changed 

## How user-item interactions are represented
- Graph-Aggregator
- there is user embedding and item embedding
- the user embedding is embedding that is created from items he interacted with
- the item embeddings are created from users they interacted with
- LightGCN-style aggregation - weighted average of neighboring embeddings (items that are interacted with)

## Start of trainig
- user embeddings are initialized to random 
- item embeddings are semantic embeddings from LLM

## Training Phase
- you cannot immediately change semantical items with random user embeddings - that will add randomness
- Stage 1: Semantic Aligning Phase: Keep items stable, and move users into the semantic item space. 
- Stage 2: Collaborative Refining Phase: Freeze the now-good user representations, and carefully refine item representations with collaborative information.

## Semantic align phase 
- now users have meaningfull semantic representation and I need them to align their embeddings to the item embeddings space - **moving user representations into item representation space**
- there is loss function that is defined as distance of the user embedding from all frozen initial item embeddings
- and second loss finction that penalises users crowding small space 
- the two functions are added and that is the total loss function
- the user representaions are the weights - there is no transforming neural network - you just calculate gradient and backpropagate on the embedding
- **random users → aggregator → alignment loss → gradient → update users**

## Collaborative refining phase
- flip roles - users frozen and I want to slightli move items
- I do not want to drastically change the item embeddings so I use MLP adaptor
- learned user embeddings teach the MLP adaptor how to refine the items 
- the mlp is trained on loss finction that minimizes difference between users embeddings with transformed item embeddings that they interacted with

 ## inference
 - User-item pair scored with dot product
 - thanks to training - same space so near means similar means will like


 ## What I will do
 - Project is mainly focused on the architecture and the text encoders arent studied deeply
 - I will try to study more how the instruction based encoders compare to general purpose ones 
 - and I will study how the different embedding space dimensions affect the performance