function  AA=kernel_matrix(rou,cta)
     AA=zeros(2,2);
     AA(1,1)=rou*cos(cta)^2+sin(cta)^2/rou;
     AA(1,2)=(rou-1/rou)*cos(cta)*sin(cta);
     AA(2,1)=AA(1,2);
     AA(2,2)=rou*sin(cta)^2+cos(cta)^2/rou;
     