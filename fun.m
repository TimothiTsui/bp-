function error = fun(x,inputnum,hiddennum,outputnum,net,inputn,outputn)
%该函数用于计算适应度值
%x input    个体
% inputnum input 输入层节点数
% outputnum input 输出层节点数
% net input 网络
% inputn input 训练输入数据
% outputn input 训练输出数据
% error output 个体适应度值

%BP神经网络初始权值和阈值，x为个体

w1=x(1:inputnum*hiddennum);
B1=x(inputnum*hiddennum+1:inputnum*hiddennum+hiddennum);
w2=x(inputnum*hiddennum+hiddennum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum);
B2=x(inputnum*hiddennum+hiddennum+hiddennum*outputnum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum+outputnum);

net.iw{1,1}=reshape(w1,hiddennum,inputnum);
net.lw{2,1}=reshape(w2,outputnum,hiddennum);
net.b{1}=reshape(B1,hiddennum,1);
net.b{2}=B2;

%BP神经网络构建
net.trainParam.epochs=20;
net.trainParam.lr=0.1;
net.trainParam.goal=0.00001;
net.trainParam.show=100;
net.trainParam.showWindow=0;

%BP神经网络训练
net=train(net,inputn,outputn);

%BP神经网络预测
an = sim(net,inputn);

%预测误差和作为个体适应度值
error=sum(abs(an-outputn));

end

