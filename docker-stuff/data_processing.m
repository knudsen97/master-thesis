clc;clear;close all;

% Import data folder results
%Get information about what's inside your folder.
myfiles = dir('results_grid_search/');
%Get the filenames and folders of all files and folders inside the folder
%of your choice.
filenames={myfiles(:).name}';
filefolders={myfiles(:).folder}';
%Get only those files that have a csv extension and their corresponding
%folders.
csvfiles=filenames(endsWith(filenames,'.csv'));
csvfolders=filefolders(endsWith(filenames,'.csv'));
%Make a cell array of strings containing the full file locations of the
%files.
files=fullfile(csvfolders,csvfiles);
% Skip first file (grid_search_results.csv)
files=files(2:end);

% Sort the cell array
files = natsortfiles(files);

% Define empty arrays for grid search table
total_epochs_elapsed = [];
delta_loss = [];
recalls = [];
precisions = [];
f1s = [];

title_font_size = 16;
axis_font_size = 15.5;
legend_font_size = 16;

for file_idx = 1:length(files)
    fig1 = ((file_idx-1)*3+1);
    fig2 = ((file_idx-1)*3+2);
    fig3 = ((file_idx-1)*3+3);

    filename = string(files(file_idx));

    % Open CSV file and convert to array
    T = readtable(filename, 'NumHeaderLines', 0);
    A = table2array(T);

    % Read first 2 lines which is hyperparameters
    % epochs, encoder_depth, lr, batch_size, l2_penalization
    hyperparameters = A(1,1:end-1);
    epochs = hyperparameters(1);
    encoder_depth = hyperparameters(2);
    lr = hyperparameters(3);
    batch_size = hyperparameters(4);
    l2_penalization = hyperparameters(5); % Weight decay
    A = A(3:end,:);


    % CSV file is structured like: 
    %[train_losses, test_losses, train_recall, train_precision, test_recall, teste_precision]
    train_losses    = A(:,1);
    test_losses     = A(:,2);
    train_recall    = A(:,3);
    train_precision = A(:,4);
    test_recall     = A(:,5);
    test_precision  = A(:,6);
    epochs_elapsed = length(train_losses);
    test = test_recall(end);
    % For the Grid Search Table. The one I generated in Python was
    % apparently wrong..
    total_epochs_elapsed = [total_epochs_elapsed; epochs_elapsed];
    delta_loss = [delta_loss; abs(train_losses(end) - test_losses(end))];
    recalls = [recalls; test_recall(end)];
    precisions = [precisions; test_precision(end)];
    f1s = [f1s; 2*(test_precision(end)*test_recall(end))/(test_precision(end)+test_recall(end))];

    lr1 = num2str(lr);
   
    % Plot train and test precision/recall
    figure(fig1)
    hold on
    plot(1:1:epochs_elapsed, train_losses)
    plot(1:1:epochs_elapsed, test_losses)
    title(compose('Train and Validation Losses\nGrid Search Test %d (lr=%s)', file_idx, lr1),'FontSize', title_font_size)
    legend('Train loss', 'Validation loss', 'FontSize', legend_font_size)
    xlabel('# of epochs')%, 'FontSize', axis_font_size)
    ylabel('Loss')%, 'FontSize', axis_font_size)
    set(gca,'FontSize', axis_font_size)

    hold off
    
    figure(fig2)
    hold on
    plot(1:1:epochs_elapsed, train_recall)
    plot(1:1:epochs_elapsed, test_recall)
    title(compose('Train and Validation Recall\nGrid Search Test %d (lr=%s)', file_idx, lr1), 'FontSize', title_font_size)
    legend('Train recall', 'Validation recall', 'FontSize', legend_font_size)
    xlabel('# of epochs')%, 'FontSize', axis_font_size)
    ylabel('Recall [%]')%, 'FontSize', axis_font_size)
    set(gca,'FontSize', axis_font_size)

    [minA,maxA] = bounds(train_recall);
    ylim([minA-0.05 1])
    % xlim([0 30])
    hold off
    
    figure(fig3)
    hold on
    plot(1:1:epochs_elapsed, train_precision)
    plot(1:1:epochs_elapsed, test_precision)
    title(compose('Train and Validation Precision\nGrid Search Test %d (lr=%s)', file_idx, lr1), 'FontSize', title_font_size)
    legend('Train precision', 'Validation precision', 'FontSize', legend_font_size)
    xlabel('# of epochs')%, 'FontSize', axis_font_size)
    ylabel('Precision [%]')%, 'FontSize', axis_font_size)
    set(gca,'FontSize', axis_font_size)

    [minA,maxA] = bounds(train_precision);
    ylim([minA-0.05 1])
    % xlim([0 30])
    hold off
    drawnow
    str1 = string(compose('figures_grid_search/loss_%d.pdf', file_idx));
    str2 = string(compose('figures_grid_search/recall_%d.pdf', file_idx));
    str3 = string(compose('figures_grid_search/precision_%d.pdf', file_idx));
    
    exportgraphics(figure(fig1), str1, 'BackgroundColor', 'none')
    exportgraphics(figure(fig2), str2, 'BackgroundColor', 'none')
    exportgraphics(figure(fig3), str3, 'BackgroundColor', 'none')
    close all

end
total_epochs_elapsed

max(recalls)
max(precisions)
max(f1s)
min(delta_loss)

% recalls
% precisions
% f1s
% delta_loss

%% Read single file
clc;clear;close all;

title_font_size = 16;
axis_font_size = 15.5;
legend_font_size = 16;

print = 0;

header = 'Fine-Tuning';

file_idx = '10.3';
id = '10_all_reg';
filename = append('results/results_', id, '.csv');

% Open CSV file and convert to array
T = readtable(filename, 'NumHeaderLines', 0);
A = table2array(T);

% Read first 2 lines which is hyperparameters
% epochs, encoder_depth, lr, batch_size, l2_penalization
hyperparameters = A(1,1:end-1);
epochs = hyperparameters(1);
encoder_depth = hyperparameters(2);
lr = hyperparameters(3);
lr1 = num2str(lr);
batch_size = hyperparameters(4);
l2_penalization = hyperparameters(5); % Weight decay
A = A(3:end,:);

% Define empty arrays for grid search table
total_epochs_elapsed = [];
delta_loss = [];
recalls = [];
precisions = [];
f1s = [];


% CSV file is structured like: 
%[train_losses, test_losses, train_recall, train_precision, test_recall, teste_precision]
train_losses    = A(:,1);
test_losses     = A(:,2);
train_recall    = A(:,3);
train_precision = A(:,4);
test_recall     = A(:,5);
test_precision  = A(:,6);
epochs_elapsed = length(train_losses);
test = test_recall(end);

% For the Grid Search Table. The one I generated in Python was
% apparently wrong..
total_epochs_elapsed = [total_epochs_elapsed; epochs_elapsed];
delta_loss = [delta_loss; abs(train_losses(end) - test_losses(end))];
recalls = [recalls; test_recall(end)];
precisions = [precisions; test_precision(end)];
f1s = [f1s; 2*(test_precision(end)*test_recall(end))/(test_precision(end)+test_recall(end))];

% Plot train and test precision/recall
figure(1)
hold on
plot(1:1:epochs_elapsed, train_losses)
plot(1:1:epochs_elapsed, test_losses)
title(compose('Train and Validation Losses\n Model %s %s (lr=%s)', file_idx, header, lr1), 'FontSize', title_font_size)

legend('Train loss', 'Validation loss', 'FontSize', legend_font_size)
xlabel('# of epochs')
ylabel('Loss')
set(gca,'FontSize', axis_font_size)

hold off

figure(2)
hold on
plot(1:1:epochs_elapsed, train_recall)
plot(1:1:epochs_elapsed, test_recall)
title(compose('Train and Validation Recall\nModel %s %s (lr=%s)', file_idx, header, lr1), 'FontSize', title_font_size)
legend('Train recall', 'Validation recall', 'FontSize', legend_font_size)
xlabel('# of epochs')
ylabel('Recall [%]')
set(gca,'FontSize', axis_font_size)

[minA,maxA] = bounds(train_recall);
ylim([minA-0.05 1.01])
% xlim([0 30])
hold off

figure(3)
hold on
plot(1:1:epochs_elapsed, train_precision)
plot(1:1:epochs_elapsed, test_precision)
title(compose('Train and Validation Precision\nModel %s %s (lr=%s)', file_idx, header, lr1), 'FontSize', title_font_size)
legend('Train precision', 'Validation precision', 'FontSize', legend_font_size)

xlabel('# of epochs')
ylabel('Precision [%]')
set(gca,'FontSize', axis_font_size)

[minA,maxA] = bounds(train_precision);
ylim([minA-0.05 1.01])
% xlim([0 30])
hold off
drawnow


loss = test_losses(length(test_losses))
recall = test_recall(length(test_recall))
precision = test_precision(length(test_precision))
f1 = 2*(test_precision(end)*test_recall(end))/(test_precision(end)+test_recall(end))
dL = abs(test_losses(end) - train_losses(end))
length(test_losses)



% Export pdf's
str1 = string(compose('figures/loss_%s.pdf', id));
str2 = string(compose('figures/recall_%s.pdf', id));
str3 = string(compose('figures/precision_%s.pdf', id));

if print
    exportgraphics(figure(1), str1, 'BackgroundColor', 'none')
    exportgraphics(figure(2), str2, 'BackgroundColor', 'none')
    exportgraphics(figure(3), str3, 'BackgroundColor', 'none')
    close all
end
