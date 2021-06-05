function visualize_mts(x,labels,spacing,inrows)

%--------------------------------------------------------------------------
%
%                   VISUALIZE MULTIPLE TIME SERIES
%
% PURPOSE
% The function VISUALIZE_MTS serves to visualize multiple time series on a 
% single plot. FTime series are displayed with a vertical space that can be
% adjusted by the user.
%
% USAGE 
% visualize_mts(x,labels,spacing,inrows)
%
% INPUTS
% x:        multivariate time series 
% labels:   string or cell array of labels (default = 1,2,...)
% spacing:  space between successive time series (default = largest median 
%           absolute deviation of the time series multiplied by 3)
% inrows:   set to 'true' if time series are in rows, 'false' otherwise
%           (default = 'true')
%
%--------------------------------------------------------------------------


if ~exist('inrows','var')
    inrows = true;
end

if inrows
    x = x.';
end

[T,N] = size(x);

if ~exist('labels','var') || isempty(labels)
    labels = 1:N;
end

if ~exist('spacing','var') || isempty(spacing)
    spacing = 3 * max(mad(x,1,2));
end


for i = 1:N
    x(:,i) = x(:,i) - mean(x(:,i)) + (N-i) * spacing;
end

plot(1:T,x,'Color','k');
yticks(linspace(0,(N-1)*spacing,N));
yticklabels(labels(N:-1:1));
xlabel('Time');
ylim([min(x(:)),max(x(:))]);





    
