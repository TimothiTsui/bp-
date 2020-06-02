function error = fun(x,inputnum,hiddennum,outputnum,net,inputn,outputn)
%�ú������ڼ�����Ӧ��ֵ
%x input    ����
% inputnum input �����ڵ���
% outputnum input �����ڵ���
% net input ����
% inputn input ѵ����������
% outputn input ѵ���������
% error output ������Ӧ��ֵ

%BP�������ʼȨֵ����ֵ��xΪ����

w1=x(1:inputnum*hiddennum);
B1=x(inputnum*hiddennum+1:inputnum*hiddennum+hiddennum);
w2=x(inputnum*hiddennum+hiddennum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum);
B2=x(inputnum*hiddennum+hiddennum+hiddennum*outputnum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum+outputnum);

net.iw{1,1}=reshape(w1,hiddennum,inputnum);
net.lw{2,1}=reshape(w2,outputnum,hiddennum);
net.b{1}=reshape(B1,hiddennum,1);
net.b{2}=B2;

%BP�����繹��
net.trainParam.epochs=20;
net.trainParam.lr=0.1;
net.trainParam.goal=0.00001;
net.trainParam.show=100;
net.trainParam.showWindow=0;

%BP������ѵ��
net=train(net,inputn,outputn);

%BP������Ԥ��
an = sim(net,inputn);

%Ԥ��������Ϊ������Ӧ��ֵ
error=sum(abs(an-outputn));

end

