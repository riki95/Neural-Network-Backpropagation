function [] = plot_curve(iter, train_err, val_err, train_acc, val_acc)
    
    fontsize = 18;
    figure;
    
    if nargin > 3
        h(1) = subplot(1,2,1);
        plot(1:iter,train_acc(1:iter), 'r', 'DisplayName','Training','LineWidth',1.5); hold on;
        plot(1:iter, val_acc(1:iter),'b-.', 'DisplayName', 'Validation','LineWidth',1.5); hold off;
        legend('show');
        title('Training and validation accuracy');
        set(gca,'fontsize',fontsize);
    end
    % display errors plot
    if nargin > 3
        h(2) = subplot(1,2,2);
    else
        h(2)=subplot(1,1,1);
    end
    
    plot(1:iter,train_err(1:iter),'r', 'DisplayName', 'Training','LineWidth',1.5); hold on;
    plot(1:iter,val_err(1:iter),'b-.', 'DisplayName', 'Validation','LineWidth',1.5); hold off;
    legend('show');
    title('Training and validation errors');
    
    set(gca,'fontsize',fontsize);
end