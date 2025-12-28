function [ur,yr,Jk]=ssmpc(p,m,nu,ny,nx,nsim,q,r,A,B,C,Ap,Bp,Cp,umax,umin,dumax,yspp,uss,yss,y0,u0)
%  Simulates the closed-loop system with MPC based on a state-space model in the
%  positional form
%  Output variables:
%  ur - system optimal inputs (dimension: nu x nsim)
%  yr - system output (dimension: ny x nsim)
%  Jk - Control cost (dimension: 1 x nsim)
%  Intput variables:
%  p    - Prediction horizon
%  m    - Control horizon
%  nu   - Number of inputs
%  ny   - Number of outputs
%  nx   - Dimension of the state vector
%  nsim - Simulation time
%  q  - Output weights (dimension: 1 x ny)
%  r -  Input weights (dimension: 1 x nu)
%  A,B,C - State, input and output matrices of the state-space model used in the MPC controller
%  Ap,Bp,Cp - State, input and output matrices of the state-space model used to represent the true plant
%  umax,umin - Max and min values for the inputs (dimension: ny x 1)
%  dumax - Max input change (dimension: ny x 1)
%  ys   - Set-points for the outputs (dimension: ny x 1)
%  yss  - Steady-state of outputs (dimension: ny x 1)
%  uss  - Steady-state of inputs  (dimension: nu x 1)
%  y0   - Initial value of outputs (dimension: ny x 1)
%  u0   - Initial value of inputs  (dimension: nu x 1)

% Defining the initial conditions (deviation variables)
xpk=y0-yss ; % (dimension: nx x 1)
xmk=xpk    ; % (dimension: nx x 1)
ypk=y0-yss ; % (dimension: ny x 1)
uk_1=u0-uss; % (dimension: nu x 1)

% ysp=[];
% for i=1:p;
%   ysp=[ysp;(ys-yss)]; % set-point vector (p.ny x 1)
% end

Psi=[];ThA=[];
for in=1:p;
    Psi=[Psi;C*A^in];
    ThA=[ThA;C*A^(in-1)*B];
end

% Creating the Dynamic Matrix
a=ThA;
Dm=[a];
if m >= 2
    for iu=1:m-2;
        a=[zeros(ny,nu);a(1:(p-1)*ny,:)];
        Dm=[Dm a];
    end
    b=C*B;
    Ai=eye(nx);
    for in=1:p-m;
        Ai=Ai+A^in;
        b=[b;C*Ai*B];
    end
    Theta=[Dm [zeros(ny*(m-1),nu);b]];
end
if m==1
    b=C*B;
    Ai=eye(nx);
    for in=1:p-m;
        Ai=Ai+A^in;
        b=[b;C*Ai*B];
    end
    Theta=b;
end

% Matrices Qbar and Rbar
aux=[];
for in=1:p;
    aux=[aux q];
end
Qbar=diag(aux);

clear aux; aux=[];
for in=1:m;
    aux=[aux r];
end
Rbar=diag(aux);
% Qbar,Rbar,pause

M=[zeros((m-1)*nu,nu) eye(nu*(m-1));zeros(nu) zeros(nu,nu*(m-1))];
Ibar=[eye(nu);zeros(nu*(m-1),nu)];
IM=eye(nu*m)-M';
% M,Ibar,IM,pause

%Matrix H
H=Theta'*Qbar*Theta+IM'*Rbar*IM;
H=(H+H')/2;

% Auxiliary constraint matrix
Dumax=dumax;
Umax=umax-uss;
Umin=umin-uss;
for i=1:m-1;
    Umax=[Umax;umax-uss];
    Umin=[Umin;umin-uss];
    Dumax=[Dumax;dumax];
end

% Starting simulation
for in=1:nsim
    ur(:,in)=uk_1+uss;
    yr(:,in)=ypk+yss;
    if in <= 20
        ys = [0; 0; 0];
    elseif in <= 80
        ys = [1.33809828; -5.23812909; 4.01255568];
    elseif in <= 140
        ys = [0.48037208; -1.57549281; 1.85025552];
    else
        ys = [1.3533336; -2.90826548; 4.64134915];
    end
    ysp=[];
    for i=1:p;
        ysp=[ysp;(ys-yss)]; % set-point vector (p.ny x 1)
    end
    el = Psi*xmk-ysp;
    ct = el'*Qbar*Theta-uk_1'*Ibar'*Rbar*IM;
    c = (Psi*xmk-ysp)'*Qbar*(Psi*xmk-ysp)+uk_1'*Ibar'*Rbar*Ibar*uk_1;

    % Including constraints on the input changes
    Ain=[IM;-IM];
    Bin=[Dumax+Ibar*uk_1;Dumax-Ibar*uk_1];
    options=optimoptions('quadprog','display','off');
    ukk=quadprog(H,ct,Ain,Bin,[],[],Umin,Umax,[],options);
    uk=ukk(1:nu); % receding horizon
    Jk(in)=ukk'*H*ukk+2*ct*ukk+c;

    % Correction of the last control input
    xmk=A*xmk+B*uk;
    ymk=C*xmk;
    if in>=101
        xpk=Ap*xpk+Bp*(uk);
        ypk=Cp*xpk;
    else
        xpk=Ap*xpk+Bp*(uk);
        ypk=Cp*xpk;
    end
    uk_1=uk;
end