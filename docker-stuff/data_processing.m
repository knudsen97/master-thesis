clc;clear;close all;

% Import data folder results
%Get information about what's inside your folder.
myfiles = dir('results/');
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

filename = string(files(2))

% Open CSV file and convert to array
T = readtable(filename, 'NumHeaderLines', 0);
A = table2array(T);

% Read first 2 lines which is hyperparameters
% epochs, encoder_depth, lr, batch_size, l2_penalization
hyperparameters = A(1,1:end-1)
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

% Plot train and test precision/recall
figure(1)
hold on
plot(1:1:epochs, train_losses)
plot(1:1:epochs, test_losses)
title('Train and test losses for baseline model with regularzation')
legend('Train loss', 'Test loss')
xlabel('# of epochs')
ylabel('Loss')
hold off

figure(2)
hold on
plot(1:1:epochs, train_recall)
plot(1:1:epochs, test_recall)
title('Train and test recall for baseline model with regularzation')
legend('Train recall', 'Test recall')
xlabel('# of epochs')
ylabel('Recall [%]')
[minA,maxA] = bounds(train_recall);
ylim([minA-0.05 1])
xlim([0 100])
hold off

figure(3)
hold on
plot(1:1:epochs, train_precision)
plot(1:1:epochs, test_precision)
title('Train and test precision for baseline model with regularzation')
legend('Train precision', 'Test precision')

xlabel('# of epochs')
ylabel('Precision [%]')
[minA,maxA] = bounds(train_precision);
ylim([minA-0.05 1])
xlim([0 100])
hold off


exportgraphics(figure(1), 'figures/loss_1.pdf', 'BackgroundColor', 'none')
exportgraphics(figure(2), 'figures/recall_1.pdf', 'BackgroundColor', 'none')
exportgraphics(figure(3), 'figures/precision_1.pdf', 'BackgroundColor', 'none')




