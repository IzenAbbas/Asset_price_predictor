library;

import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../models/prediction_model.dart';
import '../services/api_service.dart';
import '../theme/app_theme.dart';

class CarPredictionScreen extends StatefulWidget {
  final DropdownOptions options;
  const CarPredictionScreen({super.key, required this.options});

  @override
  State<CarPredictionScreen> createState() => _CarPredictionScreenState();
}

class _CarPredictionScreenState extends State<CarPredictionScreen>
    with SingleTickerProviderStateMixin {
  final _formKey = GlobalKey<FormState>();
  final _api = ApiService();

  final _urlCtrl = TextEditingController();
  final _modelYearCtrl = TextEditingController(text: '2020');
  final _mileageCtrl = TextEditingController(text: '50000');
  final _engineCapCtrl = TextEditingController(text: '1300');
  final _brandCtrl = TextEditingController();
  final _modelNameCtrl = TextEditingController();

  String? _fuelType;
  String? _transmission;
  String? _assembly;

  bool _isLoading = false;
  bool _isExtracting = false;
  CarPredictionOutput? _result;
  String? _error;
  String? _extractionError;

  late AnimationController _animCtrl;
  late Animation<double> _fadeAnim;

  @override
  void initState() {
    super.initState();
    _animCtrl = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 600),
    );
    _fadeAnim = CurvedAnimation(parent: _animCtrl, curve: Curves.easeOut);

    final car = widget.options.car;
    _fuelType = car['fuel_types']?.firstOrNull;
    _transmission = car['transmissions']?.firstOrNull;
    _assembly = car['assemblies']?.firstOrNull;
    _brandCtrl.text = car['brands']?.firstOrNull ?? '';
    if (_brandCtrl.text.isNotEmpty &&
        widget.options.carBrandModels.containsKey(_brandCtrl.text)) {
      _modelNameCtrl.text =
          widget.options.carBrandModels[_brandCtrl.text]?.firstOrNull ?? '';
    } else {
      _modelNameCtrl.text = '';
    }
  }

  @override
  void didUpdateWidget(covariant CarPredictionScreen oldWidget) {
    super.didUpdateWidget(oldWidget);

    final car = widget.options.car;
    if (_brandCtrl.text.trim().isEmpty) {
      _brandCtrl.text = car['brands']?.firstOrNull ?? '';
    }
    if (_modelNameCtrl.text.trim().isEmpty) {
      if (_brandCtrl.text.isNotEmpty &&
          widget.options.carBrandModels.containsKey(_brandCtrl.text)) {
        _modelNameCtrl.text =
            widget.options.carBrandModels[_brandCtrl.text]?.firstOrNull ?? '';
      }
    }
  }

  @override
  void dispose() {
    _animCtrl.dispose();
    _urlCtrl.dispose();
    _modelYearCtrl.dispose();
    _mileageCtrl.dispose();
    _engineCapCtrl.dispose();
    _brandCtrl.dispose();
    _modelNameCtrl.dispose();
    super.dispose();
  }

  Future<void> _extractVehicleFields() async {
    final url = _urlCtrl.text.trim();
    if (url.isEmpty) {
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(const SnackBar(content: Text('Please enter a URL')));
      return;
    }

    setState(() {
      _isExtracting = true;
      _extractionError = null;
    });

    try {
      final extracted = await _api.extractVehicleFields(url);

      if (!mounted) return;

      setState(() {
        _isExtracting = false;
        _extractionError = null;

        if (extracted.modelYear != null) {
          _modelYearCtrl.text = extracted.modelYear.toString();
        }
        if (extracted.mileageKm != null) {
          _mileageCtrl.text = extracted.mileageKm!.toStringAsFixed(0);
        }
        if (extracted.engineCapacityCc != null) {
          _engineCapCtrl.text = extracted.engineCapacityCc!.toStringAsFixed(0);
        }
        if (extracted.fuelType != null &&
            widget.options.car['fuel_types']!.contains(extracted.fuelType)) {
          _fuelType = extracted.fuelType;
        }
        if (extracted.transmission != null &&
            widget.options.car['transmissions']!.contains(
              extracted.transmission,
            )) {
          _transmission = extracted.transmission;
        }
        if (extracted.assembly != null &&
            widget.options.car['assemblies']!.contains(extracted.assembly)) {
          _assembly = extracted.assembly;
        }
        if (extracted.brand != null) {
          _brandCtrl.text = extracted.brand!;
        }
        if (extracted.modelName != null) {
          _modelNameCtrl.text = extracted.modelName!;
        }
      });

      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Vehicle details extracted and populated!'),
          duration: Duration(seconds: 2),
        ),
      );
    } catch (e) {
      setState(() {
        _isExtracting = false;
        _extractionError = ApiService.userMessageFromError(
          e,
          fallback:
              'We could not read details from that listing. You can fill the form manually.',
        );
      });
    }
  }

  Future<void> _predict() async {
    if (!_formKey.currentState!.validate()) return;

    setState(() {
      _isLoading = true;
      _error = null;
      _result = null;
    });
    _animCtrl.reset();

    try {
      final input = CarPredictionInput(
        modelYear: int.parse(_modelYearCtrl.text),
        mileage: double.parse(_mileageCtrl.text),
        engineCapacity: double.parse(_engineCapCtrl.text),
        fuelType: _fuelType!,
        transmission: _transmission!,
        assembly: _assembly!,
        brand: _brandCtrl.text.trim(),
        modelName: _modelNameCtrl.text.trim(),
      );
      final output = await _api.predictCarPrice(input);
      setState(() {
        _result = output;
        _isLoading = false;
      });
      _animCtrl.forward();
    } catch (e) {
      setState(() {
        _error = ApiService.userMessageFromError(
          e,
          fallback:
              'We could not generate a car price right now. Please try again.',
        );
        _isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final car = widget.options.car;
    final scheme = Theme.of(context).colorScheme;
    final isDark = Theme.of(context).brightness == Brightness.dark;

    return SingleChildScrollView(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      child: Form(
        key: _formKey,
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Row(
              children: [
                Container(
                  width: 40,
                  height: 40,
                  decoration: BoxDecoration(
                    color: scheme.primary.withValues(alpha: 0.12),
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Icon(
                    Icons.directions_car_rounded,
                    color: scheme.primary,
                    size: 22,
                  ),
                ),
                const SizedBox(width: 12),
                Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Car Price Predictor',
                      style: Theme.of(context).textTheme.titleLarge?.copyWith(
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                    Text(
                      'Estimate used car value in Pakistan',
                      style: Theme.of(context).textTheme.bodySmall?.copyWith(
                        color: isDark
                            ? AppColors.textSecondaryDark
                            : AppColors.textSecondaryLight,
                      ),
                    ),
                  ],
                ),
              ],
            ),
            const SizedBox(height: 20),

            GlassCard(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      const Icon(
                        Icons.link_rounded,
                        color: AppColors.primary,
                        size: 18,
                      ),
                      const SizedBox(width: 8),
                      Text(
                        'Auto-fill from Listing',
                        style: Theme.of(context).textTheme.titleSmall?.copyWith(
                          fontWeight: FontWeight.w600,
                          color: AppColors.primary,
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 12),
                  TextFormField(
                    controller: _urlCtrl,
                    decoration: InputDecoration(
                      labelText: 'Listing URL',
                      hintText: 'e.g. https://www.pakwheels.com/... ',
                      hintStyle: const TextStyle(fontSize: 12),
                    ),
                    validator: (v) {
                      if (v == null || v.isEmpty) return null;
                      if (!v.contains('http')) return 'Enter a valid URL';
                      return null;
                    },
                  ),
                  const SizedBox(height: 12),
                  SizedBox(
                    width: double.infinity,
                    child: OutlinedButton.icon(
                      onPressed: _isExtracting ? null : _extractVehicleFields,
                      icon: _isExtracting
                          ? const SizedBox(
                              height: 18,
                              width: 18,
                              child: CircularProgressIndicator(
                                strokeWidth: 2,
                                valueColor: AlwaysStoppedAnimation<Color>(
                                  AppColors.primary,
                                ),
                              ),
                            )
                          : const Icon(Icons.auto_fix_high),
                      label: Text(
                        _isExtracting ? 'Extracting...' : 'Extract Details',
                      ),
                      style: OutlinedButton.styleFrom(
                        padding: const EdgeInsets.symmetric(vertical: 14),
                        foregroundColor: AppColors.primary,
                        side: BorderSide(
                          color: AppColors.primary.withValues(alpha: 0.45),
                        ),
                      ),
                    ),
                  ),
                  if (_extractionError != null) ...[
                    const SizedBox(height: 12),
                    Container(
                      padding: const EdgeInsets.all(12),
                      decoration: BoxDecoration(
                        color: AppColors.error.withValues(alpha: 0.08),
                        borderRadius: BorderRadius.circular(12),
                        border: Border.all(
                          color: AppColors.error.withValues(alpha: 0.18),
                        ),
                      ),
                      child: Row(
                        children: [
                          const Icon(
                            Icons.error_outline,
                            color: AppColors.error,
                            size: 18,
                          ),
                          const SizedBox(width: 10),
                          Expanded(
                            child: Text(
                              _extractionError!,
                              style: Theme.of(context).textTheme.bodySmall
                                  ?.copyWith(color: AppColors.error),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ],
                ],
              ),
            ),
            const SizedBox(height: 20),

            GlassCard(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  _sectionTitle('Vehicle Details'),
                  const SizedBox(height: 16),
                  _numField(
                    _modelYearCtrl,
                    'Model Year',
                    'e.g. 2020',
                    isInt: true,
                  ),
                  const SizedBox(height: 14),
                  _numField(_mileageCtrl, 'Mileage (km)', 'e.g. 50000'),
                  const SizedBox(height: 14),
                  _numField(
                    _engineCapCtrl,
                    'Engine Capacity (cc)',
                    'e.g. 1300',
                  ),
                ],
              ),
            ),
            const SizedBox(height: 16),

            GlassCard(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  _sectionTitle('Specifications'),
                  const SizedBox(height: 16),
                  _dropdown(
                    'Fuel Type',
                    car['fuel_types'] ?? [],
                    _fuelType,
                    (v) => setState(() => _fuelType = v),
                  ),
                  const SizedBox(height: 14),
                  _dropdown(
                    'Transmission',
                    car['transmissions'] ?? [],
                    _transmission,
                    (v) => setState(() => _transmission = v),
                  ),
                  const SizedBox(height: 14),
                  _dropdown(
                    'Assembly',
                    car['assemblies'] ?? [],
                    _assembly,
                    (v) => setState(() => _assembly = v),
                  ),
                  const SizedBox(height: 14),
                  _textLookupField(
                    label: 'Brand',
                    controller: _brandCtrl,
                    options: car['brands'] ?? [],
                    onChanged: (newBrand) {
                      setState(() {
                        _modelNameCtrl.text = '';
                      });
                    },
                  ),
                  const SizedBox(height: 14),
                  Builder(
                    builder: (context) {
                      final selectedBrand = _brandCtrl.text.trim();
                      final hasBrand = selectedBrand.isNotEmpty;
                      final modelOptions =
                          hasBrand &&
                              widget.options.carBrandModels.containsKey(
                                selectedBrand,
                              )
                          ? widget.options.carBrandModels[selectedBrand]!
                          : <String>[];

                      return _textLookupField(
                        label: 'Model Name',
                        controller: _modelNameCtrl,
                        options: modelOptions,
                        enabled: hasBrand,
                        hintText: hasBrand
                            ? 'Select from list'
                            : 'Select a brand first',
                      );
                    },
                  ),
                  const SizedBox(height: 8),
                  Text(
                    'Loaded ${car['brands']?.length ?? 0} brands.',
                    style: GoogleFonts.inter(
                      fontSize: 11,
                      color: Theme.of(context).brightness == Brightness.dark
                          ? AppColors.textSecondaryDark
                          : AppColors.textSecondaryLight,
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 24),

            _predictButton(),
            const SizedBox(height: 20),

            if (_result != null) _resultCard(),
            if (_error != null) _errorCard(),

            const SizedBox(height: 40),
          ],
        ),
      ),
    );
  }


  Widget _sectionTitle(String title) {
    return Text(
      title,
      style: Theme.of(context).textTheme.titleSmall?.copyWith(
        fontWeight: FontWeight.w600,
        color: Theme.of(context).colorScheme.primary,
      ),
    );
  }

  Widget _numField(
    TextEditingController ctrl,
    String label,
    String hint, {
    bool isInt = false,
  }) {
    return TextFormField(
      controller: ctrl,
      keyboardType: isInt
          ? TextInputType.number
          : const TextInputType.numberWithOptions(decimal: true),
      decoration: InputDecoration(labelText: label, hintText: hint),
      validator: (v) {
        if (v == null || v.isEmpty) return 'Required';
        if (isInt && int.tryParse(v) == null) return 'Enter a valid integer';
        if (!isInt && double.tryParse(v) == null) return 'Enter a valid number';
        return null;
      },
    );
  }

  Widget _dropdown(
    String label,
    List<String> items,
    String? value,
    ValueChanged<String?> onChanged,
  ) {
    return DropdownButtonFormField<String>(
      initialValue: items.contains(value) ? value : null,
      isExpanded: true,
      dropdownColor: Theme.of(context).colorScheme.surface,
      decoration: InputDecoration(labelText: label),
      items: items
          .map((e) => DropdownMenuItem(value: e, child: Text(e)))
          .toList(),
      onChanged: onChanged,
      validator: (v) => v == null ? 'Please select' : null,
    );
  }

  Widget _textLookupField({
    required String label,
    required TextEditingController controller,
    required List<String> options,
    String? hintText,
    bool enabled = true,
    void Function(String)? onChanged,
  }) {
    return TextFormField(
      controller: controller,
      readOnly: true,
      enabled: enabled,
      onTap: (!enabled || options.isEmpty)
          ? null
          : () => _openOptionPicker(label, options, controller, onChanged),
      decoration: InputDecoration(
        labelText: label,
        hintText: hintText ?? 'Select from list',
        suffixIcon: const Icon(Icons.arrow_drop_down_rounded),
      ),
      validator: (v) {
        if (v == null || v.trim().isEmpty) {
          return 'Required';
        }
        return null;
      },
    );
  }

  Future<void> _openOptionPicker(
    String title,
    List<String> options,
    TextEditingController targetCtrl,
    void Function(String)? onChanged,
  ) async {
    final searchCtrl = TextEditingController();

    await showModalBottomSheet<void>(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      barrierColor: Colors.black.withValues(alpha: 0.45),
      builder: (context) {
        final screenHeight = MediaQuery.of(context).size.height;

        return Padding(
          padding: const EdgeInsets.fromLTRB(12, 12, 12, 12),
          child: SafeArea(
            top: false,
            child: ClipRRect(
              borderRadius: BorderRadius.circular(24),
              child: Container(
                constraints: BoxConstraints(maxHeight: screenHeight * 0.82),
                decoration: BoxDecoration(
                  color: Theme.of(context).colorScheme.surface,
                  borderRadius: BorderRadius.circular(24),
                  border: Border.all(
                    color: Theme.of(context).brightness == Brightness.dark
                        ? AppColors.borderDark
                        : AppColors.borderLightTheme,
                  ),
                ),
                child: StatefulBuilder(
                  builder: (context, setSheetState) {
                    final q = searchCtrl.text.trim().toLowerCase();
                    final filtered = q.isEmpty
                        ? options
                        : options
                              .where((e) => e.toLowerCase().contains(q))
                              .toList();

                    return Padding(
                      padding: const EdgeInsets.fromLTRB(16, 6, 16, 16),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            title,
                            style: GoogleFonts.inter(
                              fontSize: 16,
                              fontWeight: FontWeight.w700,
                              color:
                                  Theme.of(context).brightness ==
                                      Brightness.dark
                                  ? AppColors.textPrimaryDark
                                  : AppColors.textPrimaryLight,
                            ),
                          ),
                          const SizedBox(height: 6),
                          Text(
                            'Select one option',
                            style: GoogleFonts.inter(
                              fontSize: 12,
                              color:
                                  Theme.of(context).brightness ==
                                      Brightness.dark
                                  ? AppColors.textSecondaryDark
                                  : AppColors.textSecondaryLight,
                            ),
                          ),
                          const SizedBox(height: 14),
                          TextField(
                            controller: searchCtrl,
                            decoration: const InputDecoration(
                              hintText: 'Search options',
                              prefixIcon: Icon(Icons.search),
                            ),
                            onChanged: (_) => setSheetState(() {}),
                          ),
                          const SizedBox(height: 12),
                          Text(
                            '${filtered.length} results',
                            style: GoogleFonts.inter(
                              fontSize: 12,
                              color:
                                  Theme.of(context).brightness ==
                                      Brightness.dark
                                  ? AppColors.textSecondaryDark
                                  : AppColors.textSecondaryLight,
                            ),
                          ),
                          const SizedBox(height: 8),
                          Expanded(
                            child: filtered.isEmpty
                                ? Center(
                                    child: Text(
                                      'No matching options',
                                      style: GoogleFonts.inter(
                                        fontSize: 13,
                                        color:
                                            Theme.of(context).brightness ==
                                                Brightness.dark
                                            ? AppColors.textSecondaryDark
                                            : AppColors.textSecondaryLight,
                                      ),
                                    ),
                                  )
                                : ListView.separated(
                                    itemCount: filtered.length,
                                    separatorBuilder: (context, index) =>
                                        Divider(
                                          height: 1,
                                          color:
                                              Theme.of(context).brightness ==
                                                  Brightness.dark
                                              ? AppColors.borderDark
                                              : AppColors.borderLightTheme,
                                        ),
                                    itemBuilder: (context, index) {
                                      final item = filtered[index];
                                      return ListTile(
                                        contentPadding: EdgeInsets.zero,
                                        title: Text(
                                          item,
                                          style: TextStyle(
                                            color:
                                                Theme.of(context).brightness ==
                                                    Brightness.dark
                                                ? AppColors.textPrimaryDark
                                                : AppColors.textPrimaryLight,
                                          ),
                                        ),
                                        trailing: Icon(
                                          Icons.chevron_right_rounded,
                                          color:
                                              Theme.of(context).brightness ==
                                                  Brightness.dark
                                              ? AppColors.textSecondaryDark
                                              : AppColors.textSecondaryLight,
                                        ),
                                        onTap: () {
                                          targetCtrl.text = item;
                                          if (onChanged != null) {
                                            onChanged(item);
                                          }
                                          Navigator.of(context).pop();
                                        },
                                      );
                                    },
                                  ),
                          ),
                        ],
                      ),
                    );
                  },
                ),
              ),
            ),
          ),
        );
      },
    );

    searchCtrl.dispose();
  }

  Widget _predictButton() {
    return SizedBox(
      width: double.infinity,
      child: FilledButton.icon(
        onPressed: _isLoading ? null : _predict,
        style: FilledButton.styleFrom(
          padding: const EdgeInsets.symmetric(vertical: 16),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(14),
          ),
        ),
        icon: _isLoading
            ? const SizedBox(
                height: 18,
                width: 18,
                child: CircularProgressIndicator(
                  color: Colors.white,
                  strokeWidth: 2.2,
                ),
              )
            : const Icon(Icons.auto_awesome, size: 20),
        label: Text(_isLoading ? 'Predicting...' : 'Predict Car Price'),
      ),
    );
  }

  Widget _resultCard() {
    final isDark = Theme.of(context).brightness == Brightness.dark;

    return FadeTransition(
      opacity: _fadeAnim,
      child: Container(
        decoration: BoxDecoration(
          color: Theme.of(context).colorScheme.surface,
          borderRadius: BorderRadius.circular(16),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withValues(alpha: isDark ? 0.2 : 0.04),
              blurRadius: 18,
              offset: const Offset(0, 8),
            ),
          ],
          border: Border.all(
            color: isDark ? AppColors.borderDark : AppColors.borderLightTheme,
          ),
        ),
        padding: const EdgeInsets.all(24),
        child: Column(
          children: [
            Icon(
              Icons.check_circle_rounded,
              color: Theme.of(context).colorScheme.primary,
              size: 42,
            ),
            const SizedBox(height: 12),
            Text(
              'Predicted Price',
              style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                fontWeight: FontWeight.w600,
                color: isDark
                    ? AppColors.textSecondaryDark
                    : AppColors.textSecondaryLight,
              ),
            ),
            const SizedBox(height: 6),
            Text(
              _result!.formattedPrice,
              style: Theme.of(context).textTheme.displaySmall?.copyWith(
                fontWeight: FontWeight.w800,
                color: isDark
                    ? AppColors.textPrimaryDark
                    : AppColors.textPrimaryLight,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _errorCard() {
    return Container(
      decoration: BoxDecoration(
        color: AppColors.error.withValues(alpha: 0.08),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: AppColors.error.withValues(alpha: 0.18)),
      ),
      padding: const EdgeInsets.all(16),
      child: Row(
        children: [
          const Icon(
            Icons.info_outline_rounded,
            color: AppColors.error,
            size: 24,
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Text(
              _error!,
              style: Theme.of(
                context,
              ).textTheme.bodyMedium?.copyWith(color: AppColors.error),
            ),
          ),
        ],
      ),
    );
  }
}
