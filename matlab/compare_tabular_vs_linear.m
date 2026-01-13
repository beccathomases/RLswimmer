function compare_tabular_vs_linear()
%COMPARE_TABULAR_VS_LINEAR  Run both methods and compare returns.

    nEpisodes = 500;
    Tmax      = 40;

    fprintf('Running tabular Q-learning...\n');
    returns_tab = train_swimmer_tabular(nEpisodes, Tmax, false);

    fprintf('Running linear Q-network Q-learning...\n');
    returns_lin = train_swimmer_linearQ(nEpisodes, Tmax, false);

    % Simple plotting side-by-side
    figure;
    plot(1:nEpisodes, returns_tab, '-', 'DisplayName', 'Tabular'); hold on;
    plot(1:nEpisodes, returns_lin, '--', 'DisplayName', 'Linear Q-net');
    xlabel('Episode');
    ylabel('Return (cumulative reward)');
    title('2-paddle swimmer: Tabular vs Linear Q-learning');
    legend('Location', 'best');
    grid on;
end
