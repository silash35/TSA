function [A,B,C,d0,Dd,F,psi,N]=opom(g,Ts)
s=tf('s');
[ny,nu]=size(g);
for i=1:ny
    for j=1:nu
        polos=pole(g(i,j));
        l=length(polos);
        m(i,j)=l;
    end
end
na=max(max(m)); % maximum order of stable poles
h =  g/s; % transfer function after of step application

for i=1:ny
    for j=1:nu
        [num,den]=tfdata(h(i,j),'v');
        if num==0
            d=0; r=0;
        else
            [d,r]=residue(num,den);
        end
        % matrix D0
        d0(i,j)=d(end);
        % matrix Dd
        l=length(d);
        ddaux=d(1:l-1);
        ddaux=[ddaux;zeros(na-numel(ddaux),1)];
        dd(:,j)=ddaux;
        % matrix F
        n=length(r);
        daux=r(1:n-1);
        faux=exp(daux*Ts);
        faux=[faux;zeros(na-numel(faux),1)];
        f(:,j)=faux;
    end
    Dd(:,:,i)=dd;
    F(:,:,i) =f ;
end

Ddd=Dd(:);Dd=diag(Ddd);
Fd=F(:); F=diag(Fd);

J=[];
for i=1:nu
    aux=[zeros(na,i-1) ones(na,1) zeros(na,(nu-i))];
    J=[J;aux];
end

clear aux;
phi=ones(1,nu*na);
N=[];
psi=[];
for i=1:ny
    N=[N;J];
    aux=[zeros(1,(i-1)*na*nu) phi zeros(1,(ny-i)*na*nu)];
    psi=[psi;aux];
end
nd=na*nu*ny;
A=[eye(ny) zeros(ny,nd)
   zeros(nd,ny) F]; 
B = [d0 ; Dd*F*N];
C = [eye(ny) psi];

