% function [A,B,C,F,psi]=opom_stds(g,Ts)
% Funcao para construir as matrizes do modelo OPOM em espaco de estados
% para sistemas estaveis com tempos mortos.
clear all
close all
clc
s=tf('s');
Ts=1;
% g=1/(22.8*s+1)*[0.7868 -0.6147
%                 0.8098 -0.9820];
% g=[-0.2623/(60*s^2+59.2*s+1)     0.1368/(1164*s^2+99.7*s+1)
%     0.1242/(218.7*s^2+16.2*s+1) -0.1351/(70*s^2+20*s+1)     ];
g=[-4.05/(50*s+1)   1.77/(60*s+1)
    5.39/(50*s+1)   5.72/(60*s+1)     ];
g.iodelay=[27 28;18 14];
tempo_morto=g.iodelay/Ts; % matrix of time delays in the discret time domain
% tempo_morto_maximo=max(max(g.iodelay)); % maximum dead-time in the continuous time domain
p = max(max(tempo_morto)); % maximum dead-time

[ny,nu]=size(g);
for i=1:ny
    for j=1:nu
        polos=pole(g(i,j));
        l=length(polos);
        m(i,j)=l;
    end
end
na=max(max(m)); % maximum order of stable poles
h=g/s; % transfer function after applying step on the inputs
nd=ny*nu*na;
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

% criando as matrizes Bs e Bd:
% inicialmente construindo a matriz Bs (de 0 ate o tempo morto m�ximo)
for x=1:nu       % varredura em x e y (linha e coluna)
    for y=1:ny
        for t=0:p  % varredura em l
            if (t==tempo_morto(y,x))      % se t for igual a tempo morto deste sistema SISO
                Bs(y,x,t+1)=d0(y,x);   % carrega o ganho
            else
                Bs(y,x,t+1)=0;        % do contrario, zera o elemento da matriz
            end
            
        end
    end
end

% Organizando as matrizes Dd em funcao dos tempos mortos de cada par (i,j)
ddd=zeros(nd,nd,p+1);
for i=1:ny
    for j=1:nu
        for k=0:p
            if tempo_morto(i,j)==k
                
                ddd(:,1+(i-1)*na*nu+(j-1)*na:(i-1)*na*nu+(j-1)*na+na,k+1) = Dd(:,1+(i-1)*na*nu+(j-1)*na:(i-1)*na*nu+(j-1)*na+na);
                
                % A primeira saida e primeira entrada: 1:na colunas em ddd
                % Primeira saida e segunda entrada: na+1:2na colunas em ddd
                % ...
                % Primeira saida e ultima entrada: (nu-1)*na:nu*na
                % Segunda saida e primeira entrada: nu*na+1:nu*na+na
                % Genericamente, para saida i e entrada j: 1+(i-1)*na*nu+(j-1)*na: (i-1)*na*nu+(j-1)*na + na
            end
        end
    end
end

% construindo as matrizes J e N
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

% construindo matriz Bd para cada tempo morto l, multiplicar ddd*F*N
for k=0:p
    Bd(:,:,k+1)=ddd(:,:,k+1)*F*N;
end

clear aux
aux=[];
for k=2:p+1
    aux=[aux [Bs(:,:,k); Bd(:,:,k)]];
end

A=[eye(ny) zeros(ny,nd)
   zeros(nd,ny) F]; 
B = [d0 ; Dd*F*N];
C = [eye(ny) psi];

if p>0
    % reescrevendo as matrizes A, B e C
    aux2=[[zeros(nu,(p-1)*nu);eye((p-1)*nu)] zeros(p*nu,nu)];
    A = [A aux; zeros(p*nu,ny+nd) aux2];
    B = [Bs(:,:,1); Bd(:,:,1); eye(nu); zeros((p-1)*nu,nu)];
    C = [C zeros(ny,p*nu)];
else
    A=[eye(ny) zeros(ny,nd)
       zeros(nd,ny) F]; 
    B = [d0 ; Dd*F*N];
    C = [eye(ny) psi];
end


% Teste OPOM
nsim=160;
[t,ys] = step_response(g,nsim);

nx=size(A,1);
x0=zeros(nx,1);
passos=nsim/Ts;
for j=1:passos
    if j==1
        du=ones(nu,1);
    else
        du=zeros(nu,1);
    end
    xk=A*x0+B*du;
    yy(:,j)=C*x0;
    x0=xk;
end
tspan=0:Ts:passos*Ts;
for j=1:ny
    figure(j)
    plot(tspan(1:end-1),yy(j,:),'ro',t,ys(:,j),'b-')
end