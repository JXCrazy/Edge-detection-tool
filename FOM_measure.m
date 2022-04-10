function y=FOM_measure(Ideal_edge, actual_edge)
   %Ideal_edge---ideal edge map, M*N 0-1 image
   %actual_edge---actual edge map,M*N 0-1 image
   [M,N]=size(Ideal_edge);
   Number_i=sum(Ideal_edge(:));
   Number_a=sum(actual_edge(:));
   Number_c=max(Number_i,Number_a);
   X=Ideal_edge.*actual_edge;
   A=sum(X(:));
   Y=find(Ideal_edge~=0);
   Y1=mod(Y,M)+j*(Y-mod(Y,M))/M;
   Z=find((actual_edge-X)==1);
   K=max(size(Z));
   for k=1:K
       a=mod(Z(k),M)+j*(Z(k)-mod(Z(k),M))/M;
       e=Y1-a;
       d=min(abs(e));
       A=A+1/(1+d^2/4);
   end
   y=A/Number_c;
   