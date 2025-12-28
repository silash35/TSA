function [duk,Jk]=issmpc(p,m,nu,ny,q,r,A,B,C,umax,umin,dumax,ys,uk_1,xmk)

%  Simulates the closed-loop system with MPC based on state-space model in the
%  incremental form
%  Output variables:
%  duk - system optimal inputs move (dimension: m*nu x 1)
%  Jk - Control cost (dimension: 1 x 1)
%  Intput variables:
%  p    - Optimization horizon
%  m    - Control horizon
%  nu   - Number of inputs
%  ny   - Number of outputs
%  q  - Output weights (dimension: 1 x ny)
%  r -  Input weights (dimension: 1 x nu)
%  A,B,C - State, input and output matrices of the state-space model used in the MPC controller
%  umax,umin - Max and min values for the inputs (dimension: ny x 1)
%  dumax - Max input change (dimension: ny x 1)
%  ys   - Set-points for the outputs (dimension: ny x 1)
%  uk_1 - Last value of manupulated inputs (dimension: nu x 1)
%  xmk  - current state of system (dimension: nx x 1)

ysp=[];
for i=1:p;
    ysp=[ysp;ys]; % set-point vector (p.ny x 1)
end

Phi=[];
tha=[];
for k=1:p
    Phi=[Phi; C*A^(k)];
    tha=[tha; C*A^(k-1)*B];
end
a=tha;
Dm=a;
for iu=1:m-1;
    a=[zeros(ny,nu);a(1:(p-1)*ny,:)];
    Dm=[Dm a];
end
Theta=Dm; % dimension p*ny x m*nu

%Matrices Qbar, Mtil, Itil and Rbar
aux=[];
for in=1:p;
    aux=[aux q];
end
Qbar=diag(aux);

clear aux; aux=[];

Mtil=[];
Itil=[];
auxM=zeros(nu,m*nu);
for in=1:m;
    aux=[aux r];
    auxM=[eye(nu) auxM(:,1:(m-1)*nu)];
    Mtil=[Mtil;auxM];
    Itil=[Itil;eye(nu)];
end
Rbar=diag(aux);

%Matrix H
H=Theta'*Qbar*Theta+Rbar;
H=(H+H')/2;
%Auxiliary constraint matrix
Dumax=dumax;
Umax=umax;
Umin=umin;
for i=1:m-1;
    Umax=[Umax;umax];
    Umin=[Umin;umin];
    Dumax=[Dumax;dumax];
end

% Parameters of QP
el = Phi*xmk-ysp;
ct = el'*Qbar*Theta;
c = (Phi*xmk-ysp)'*Qbar*(Phi*xmk-ysp);

% Inequality constraints Ain*x <= Bin
Ain=[Mtil;-Mtil];
Bin=[Umax-Itil*uk_1;Itil*uk_1-Umin];
options=optimoptions('quadprog','display','off');
dukk=quadprog(H,ct,Ain,Bin,[],[],-Dumax,Dumax,[],options); % optimal solution
duk=dukk(1:nu); % receding horizon
Jk=dukk'*H*dukk+2*ct*dukk+c; % control cost
