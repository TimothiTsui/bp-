function ret=select(individuals,sizepop)
%����ѡ�����
%individuals input  ��Ⱥ��Ϣ
% sizepop input ��Ⱥ��ģ
% ret output ѡ��������Ⱥ

%���ݸ�����Ӧ��ֵ��������
fitness1=10./individuals.fitness;

%����ѡ�����
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
            break;  %Ѱ����������䣬�˴�ת����ѡ����Ⱦɫ��i��ע�⣺��תsizepop�����̵Ĺ����У��п��ܻ��ظ�ѡ��ĳЩȾɫ��
        end
    end
    if length(index) == sizepop
        break;
    end
end
%����Ⱥ
individuals.chrom=individuals.chrom(index,:);
individuals.fitness=individuals.fitness(index);
ret=individuals;
end

