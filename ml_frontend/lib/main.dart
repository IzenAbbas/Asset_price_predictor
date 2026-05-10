/// Entry point for the Asset Advisor Flutter app.
///
/// Fetches dropdown options from the FastAPI backend on startup, then
/// shows a bottom-navigation shell with Car and House prediction tabs.
library;

import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:provider/provider.dart';
import 'models/prediction_model.dart';
import 'screens/car_prediction_screen.dart';
import 'screens/house_prediction_screen.dart';
import 'services/api_service.dart';
import 'theme/app_theme.dart';
import 'theme/theme_provider.dart';

void main() {
  runApp(
    ChangeNotifierProvider(
      create: (_) => ThemeProvider(),
      child: const AssetAdvisorApp(),
    ),
  );
}

class AssetAdvisorApp extends StatelessWidget {
  const AssetAdvisorApp({super.key});

  @override
  Widget build(BuildContext context) {
    final themeProvider = Provider.of<ThemeProvider>(context);
    return MaterialApp(
      title: 'Asset Advisor',
      debugShowCheckedModeBanner: false,
      theme: themeProvider.themeData,
      home: const AppShell(),
    );
  }
}

/// Top-level shell that loads dropdown options once, then renders
/// car/house tabs with a smooth bottom navigation bar.
class AppShell extends StatefulWidget {
  const AppShell({super.key});

  @override
  State<AppShell> createState() => _AppShellState();
}

class _AppShellState extends State<AppShell> {
  final _api = ApiService();
  int _currentTab = 0;

  DropdownOptions? _options;
  bool _loading = true;
  String? _error;

  @override
  void initState() {
    super.initState();
    _loadOptions();
  }

  Future<void> _loadOptions() async {
    try {
      final options = await _api.getDropdownOptions();
      setState(() {
        _options = options;
        _loading = false;
      });
    } catch (e) {
      setState(() {
        _error = e.toString().replaceFirst('Exception: ', '');
        _loading = false;
      });
    }
  }

  Future<void> _refreshOptions() async {
    setState(() {
      _loading = true;
      _error = null;
    });
    await _loadOptions();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      // ── App Bar ──
      appBar: AppBar(
        title: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(
              width: 34,
              height: 34,
              decoration: BoxDecoration(
                color: Theme.of(
                  context,
                ).colorScheme.primary.withValues(alpha: 0.12),
                borderRadius: BorderRadius.circular(10),
              ),
              child: Icon(
                Icons.analytics_rounded,
                size: 19,
                color: Theme.of(context).colorScheme.primary,
              ),
            ),
            const SizedBox(width: 10),
            Text(
              'Asset Advisor',
              style: Theme.of(
                context,
              ).textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w700),
            ),
          ],
        ),
        actions: [
          Consumer<ThemeProvider>(
            builder: (context, themeProvider, child) {
              return IconButton(
                tooltip: 'Toggle Theme',
                onPressed: () {
                  themeProvider.toggleTheme();
                },
                icon: Icon(
                  themeProvider.isDarkMode
                      ? Icons.light_mode_rounded
                      : Icons.dark_mode_rounded,
                ),
              );
            },
          ),
          IconButton(
            tooltip: 'Refresh options',
            onPressed: _refreshOptions,
            icon: const Icon(Icons.refresh_rounded),
          ),
        ],
      ),

      // ── Body ──
      body: _buildBody(),

      // ── Bottom Navigation ──
      bottomNavigationBar: _options != null
          ? Builder(
              builder: (context) {
                final isDark = Theme.of(context).brightness == Brightness.dark;
                return Container(
                  decoration: BoxDecoration(
                    color: Theme.of(context).colorScheme.surface,
                    border: Border(
                      top: BorderSide(
                        color: isDark
                            ? AppColors.borderDark
                            : AppColors.borderLightTheme,
                      ),
                    ),
                    boxShadow: [
                      BoxShadow(
                        color: isDark
                            ? Colors.black.withValues(alpha: 0.3)
                            : Colors.black.withValues(alpha: 0.05),
                        blurRadius: 16,
                        offset: const Offset(0, -4),
                      ),
                    ],
                  ),
                  child: NavigationBar(
                    backgroundColor: Colors.transparent,
                    indicatorColor: isDark
                        ? AppColors.primary.withValues(alpha: 0.2)
                        : AppColors.primary.withValues(alpha: 0.1),
                    selectedIndex: _currentTab,
                    onDestinationSelected: (i) =>
                        setState(() => _currentTab = i),
                    destinations: [
                      NavigationDestination(
                        icon: Icon(
                          Icons.directions_car_outlined,
                          color: isDark
                              ? AppColors.textSecondaryDark
                              : AppColors.textSecondaryLight,
                        ),
                        selectedIcon: Icon(
                          Icons.directions_car_rounded,
                          color: isDark
                              ? AppColors.primaryLight
                              : AppColors.primary,
                        ),
                        label: 'Car',
                      ),
                      NavigationDestination(
                        icon: Icon(
                          Icons.home_outlined,
                          color: isDark
                              ? AppColors.textSecondaryDark
                              : AppColors.textSecondaryLight,
                        ),
                        selectedIcon: Icon(
                          Icons.home_rounded,
                          color: AppColors.accent,
                        ),
                        label: 'House',
                      ),
                    ],
                  ),
                );
              },
            )
          : null,
    );
  }

  Widget _buildBody() {
    // Loading state
    if (_loading) {
      return Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const CircularProgressIndicator(color: AppColors.primaryLight),
            const SizedBox(height: 20),
            Text(
              'Connecting to prediction server...',
              style: GoogleFonts.inter(
                fontSize: 14,
                color: Theme.of(context).brightness == Brightness.dark
                    ? AppColors.textSecondaryDark
                    : AppColors.textSecondaryLight,
              ),
            ),
          ],
        ),
      );
    }

    // Error state
    if (_error != null) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(32),
          child: GlassCard(
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Icon(
                  Icons.cloud_off_rounded,
                  color: Theme.of(context).colorScheme.primary,
                  size: 48,
                ),
                const SizedBox(height: 14),
                Text(
                  'Couldn\'t load data',
                  style: Theme.of(
                    context,
                  ).textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w700),
                ),
                const SizedBox(height: 8),
                Text(
                  _error!,
                  textAlign: TextAlign.center,
                  style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                    color: Theme.of(context).brightness == Brightness.dark
                        ? AppColors.textSecondaryDark
                        : AppColors.textSecondaryLight,
                  ),
                ),
                const SizedBox(height: 20),
                FilledButton.icon(
                  onPressed: () {
                    setState(() {
                      _loading = true;
                      _error = null;
                    });
                    _loadOptions();
                  },
                  icon: const Icon(Icons.refresh_rounded),
                  label: const Text('Try again'),
                ),
              ],
            ),
          ),
        ),
      );
    }

    // Tabbed content
    return IndexedStack(
      index: _currentTab,
      children: [
        CarPredictionScreen(options: _options!),
        HousePredictionScreen(options: _options!),
      ],
    );
  }
}
