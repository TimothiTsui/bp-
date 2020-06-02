function ret=select(individuals,sizepop)
%用于选择操作
%individuals input  种群信息
% sizepop input 种群规模
% ret output 选择后的新种群

%根据个体适应度值进行排序
fitness1=10./individuals.fitness;

%个体选择概率
sumfitness=sum(fitness1);
sumf=fitness1./sumfitness;
index=[]; 
for i=1:1000
    pick=rand;
    while pick==0    
        pick=rand;        
    end
    for j=1:sizepop    
        pick=pick-sumf(mod(i,sizepop-1)+1);        
        if pick<0        
            index=[index j];            
            break;  %寻找落入的区间，此次转轮盘选中了染色体i，注意：在转sizepop次轮盘的过程中，有可能会重复选择某些染色体
        end
    end
    if length(index) == sizepop
        break;
    end
end
%新种群
individuals.chrom=individuals.chrom(index,:);
individuals.fitness=individuals.fitness(index);
ret=individuals;
end

