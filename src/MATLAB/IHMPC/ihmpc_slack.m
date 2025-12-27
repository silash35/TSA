clear all
close all
clc
s=tf('s'); % defining Laplace's variables
Ts=1; % sampling time
g = [5/(10*s+1) 8/(8*s+1)    0
     0          3.3/(15*s+1) 7/(10*s+1)
     6/(15*s+1) 0            20/(20*s+1)]; % transfer function matrix

[A,B,C,D0,Dd,F,Psi,N]=opom(g,Ts); % it creates opom model
Ap=A; Bp=B; Cp=C;
ny = size(C,1); % output variables of the system
nu = size(B,2); % input variables of the system
nx = size(A,1); % state variables of the system
nd = size(F,1); % states related to the stable pole of the system

% parameters of the IHMPC
sy = [1e2 1e2 1e2]; % weights of output slacks

nsim = 300;      % simulation time
m = 3;           % control horizon
qy = [1 1 1];      % output weights
qu = [];      % output weights
r  = 1*[1 1 1]; % move weights

umin = -[0.5 0.5 0.1]'; % lower bounds of inputs
umax =  [0.5 0.5 0.1]'; % upper bounds of inputs
dumax=  0.1*[0.5 0.5 0.1]' ; % maximum variation of input moves
% =========================================================================

% Creating the Qy, Qybar and Rbar matrices
aux1=[];
aux2=[];
aux3=[];
Qy=diag(qy); % dimension (ny x ny)
S=diag(sy);
for i=1:m
    aux1=[aux1 qy];
    %     aux2=[aux2 qu];
    aux3=[aux3 r];
end
Qybar=diag(aux1); % dimension (m*ny x m*ny)

% Qubar=diag(aux2);
Rbar=diag(aux3);  % dimension (m*nu x m*nu)
Qaux = F'*Psi'*Qy*Psi*F ;
Qbar = dlyap(F,Qaux)    ; % dimension (nd x nd)
% Qaux = Psi'*Qy*Psi ;
% Qbar = dlyap(F,Qaux)    ; % dimension (nd x nd)
Bd=Dd*F*N;
Ibar=[];
Fx=[];
blin = zeros(ny,m*nu);
Fu = [];
dlin = blin;
Dm0=[];
Futil=[];
for j=1:m
    Ibar=[Ibar;eye(ny)]; % dimension (m*ny x ny)
    Fx=[Fx;Psi*(F^j)];   % dimension (m*ny x nd)
    blin = [Psi*(F^(j-1))*Bd blin(:,1:(m-1)*nu)];
    Fu = [Fu; blin]     ; % dimension (m*ny x m*nu)
    dlin = [D0 dlin(:,1:(m-1)*nu)];
    Dm0 = [Dm0; dlin]     ; % dimension (m*ny x m*nu)
    Futil=[Futil (F^(m-j))*Bd]; % dimension (nd x m*nu)
end
Dtil0 = Dm0((m-1)*ny+1:m*ny,:); % dimension (ny x m*nu)
Itil=[];
for k=1:m
    Itil = [Itil; eye(nu)]; % dimension m*nu x nu
end

aux=Itil;
Mtil=Itil;
for j=2:m
    aux=[zeros(nu);aux(1:(m-1)*nu,:)];
    Mtil=[Mtil aux]; % dimension m*nu x nu
end

% creating the H matrix of QP
H = [(Dm0+Fu)'*Qybar*(Dm0+Fu)+Futil'*Qbar*Futil+Rbar -(Dm0+Fu)'*Qybar*Ibar
     -Ibar'*Qybar*(Dm0+Fu)                           Ibar'*Qybar*Ibar+S];

% defining the initial conditions
% =========================================================================
y0 = [1 1 1]';
u0 = [0 0 0]';
uk_1 = u0;
xmk = [y0; ones(nd,1)]; % states of model
xpk = [y0; ones(nd,1)]; % states of plant
ypk = Cp*xpk               ; % outputs of model
ysp=y0;

% =========================================================================
yspp=[];
for in=1:nsim
ur(:,in) = uk_1  ;
yr(:,in) = ypk   ;
    if in <= 200
        ysp    = [2 2 2]'    ;
%     elseif in >= 10 && in <= 199
%         ysp    = [55 1.2]'    ;
    else
        ysp    = 2.5*[1 1 1]'    ;
    end
cf = [(Ibar*xmk(1:ny)+Fx*xmk(ny+1:end)-Ibar*ysp)'*Qybar*Dm0+...
     (Ibar*xmk(1:ny)+Fx*xmk(ny+1:end)-Ibar*ysp)'*Qybar*Fu+(F^m*xmk(ny+1:end))'*Qbar*Futil ...
     (Ibar*ysp-Fx*xmk(ny+1:end)-Ibar*xmk(1:ny))'*Qybar*Ibar];
c = (Ibar*xmk(1:ny)+Fx*xmk(ny+1:end)-Ibar*ysp)'*Qybar*(Ibar*xmk(1:ny)+Fx*xmk(ny+1:end)-Ibar*ysp)+...
     (F^m*xmk(ny+1:end))'*Qbar*(F^m*xmk(ny+1:end));

% Including inequality constraints
Aineq = [ Mtil zeros(m*nu,ny)
         -Mtil zeros(m*nu,ny)];

Bineq = [ Itil*(umax-uk_1)
          Itil*(uk_1-umin) ];

% Including equality constraints
Aeq = [Dtil0 -eye(ny)];

Beq = ysp-xmk(1:ny);

% Including constraints of lower and upper bounds
UB = [Itil*dumax; ones(ny,1)*Inf];
LB = [-Itil*dumax; -ones(ny,1)*Inf];

% options = optimset('display','iter')
[dd,fvin,flagin]=quadprog(H,cf,Aineq,Bineq,Aeq,Beq,LB,UB);

% storing calculated data
fval(in) = dd'*H*dd + 2*cf*dd + c  ; % control cost value
flag(in) = flagin                  ; % exitflag of exit condition of the MATLAB's quadprog routine
duuk(:,in) = dd(1:m*nu)            ; % prediction of input moves
sky(:,in)=dd(nu*m+1:m*nu+ny)       ; % slacks of outputs
% ski(:,in)=dd(m*nu+ny+1:(m+1)*nu+ny); % slacks of inputs

duk = dd(1:nu)   ; % receding horizon
uk_1 = duk + uk_1; % inputs to be implemented in plant
xpk = Ap*xpk+Bp*duk; % states of plant
ypk = Cp*xpk      ; % output of plant
xmk=xpk; % these should be computed through of state estimator (e.g Kalman filter)
% ymk=C*xmk;
yspp=[yspp ysp];
end

nc=size(yr,1);
figure(1)
for j=1:nc
    subplot(nc,1,j)
    plot(yr(j,:),'k-')
    hold on
    plot(yspp(j,:),'r--')
    xlabel('tempo nT')
    in = num2str(j);
    yrot = ['y_' in];
    ylabel(yrot)
%     axis([0 nsim 0 4])
end
% legend('PV','set-point')

nc=size(ur,1);
figure(2)
for j=1:nc
subplot(nc,1,j)
    plot(ur(j,:),'k-')
    xlabel('tempo nT')
    in = num2str(j);
    yrot = ['u_' in];
    ylabel(yrot)
end

figure(3)
plot(fval)
xlabel('tempo nT')
ylabel('Cost function')

figure(4)
plot(flag)
xlabel('tempo nT')
ylabel('Flag')


nc=size(sky,1);
figure(5)
for j=1:nc
subplot(nc,1,j)
    plot(sky(j,:),'k-')
    xlabel('tempo nT')
    in = num2str(j);
    yrot = ['\delta_' in];
    ylabel(yrot)
end
