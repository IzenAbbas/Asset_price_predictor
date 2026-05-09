/// Car Price Prediction screen — collects car features and shows the
/// predicted price from the FastAPI backend.
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

  // Controllers
  final _urlCtrl = TextEditingController();
  final _modelYearCtrl = TextEditingController(text: '2020');
  final _mileageCtrl = TextEditingController(text: '50000');
  final _engineCapCtrl = TextEditingController(text: '1300');
  final _brandCtrl = TextEditingController();
  final _modelNameCtrl = TextEditingController();

  // Dropdown selections
  String? _fuelType;
  String? _transmission;
  String? _assembly;

  // State
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

    // Set default dropdown values
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

        // Auto-fill the form fields
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
          content: Text('✅ Vehicle details extracted and populated!'),
          duration: Duration(seconds: 2),
        ),
      );
    } catch (e) {
      final errorMsg = e.toString().replaceFirst('Exception: ', '');
      setState(() {
        _isExtracting = false;
        _extractionError = errorMsg;
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
        _error = e.toString().replaceFirst('Exception: ', '');
        _isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final car = widget.options.car;

    return SingleChildScrollView(
      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
      child: Form(
        key: _formKey,
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // ── Header ──
            Row(
              children: [
                Container(
                  padding: const EdgeInsets.all(12),
                  decoration: BoxDecoration(
                    gradient: AppGradients.primary,
                    borderRadius: BorderRadius.circular(14),
                  ),
                  child: const Icon(
                    Icons.directions_car_rounded,
                    color: Colors.white,
                    size: 28,
                  ),
                ),
                const SizedBox(width: 14),
                Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Car Price Predictor',
                      style: GoogleFonts.inter(
                        fontSize: 22,
                        fontWeight: FontWeight.w700,
                        color: Theme.of(context).brightness == Brightness.dark
                            ? AppColors.textPrimaryDark
                            : AppColors.textPrimaryLight,
                      ),
                    ),
                    Text(
                      'Estimate used car value in Pakistan',
                      style: GoogleFonts.inter(
                        fontSize: 13,
                        color: Theme.of(context).brightness == Brightness.dark
                            ? AppColors.textSecondaryDark
                            : AppColors.textSecondaryLight,
                      ),
                    ),
                  ],
                ),
              ],
            ),
            const SizedBox(height: 24),

            // ── URL Extraction Card ──
            GlassCard(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      const Icon(
                        Icons.link_rounded,
                        color: AppColors.accent,
                        size: 18,
                      ),
                      const SizedBox(width: 8),
                      Text(
                        'Auto-fill from Listing',
                        style: GoogleFonts.inter(
                          fontSize: 15,
                          fontWeight: FontWeight.w600,
                          color: AppColors.accent,
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 12),
                  TextFormField(
                    controller: _urlCtrl,
                    decoration: InputDecoration(
                      labelText: 'Listing URL',
                      hintText:
                          'e.g. https://www.pakwheels.com/... or https://www.olx.com.pk/...',
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
                    child: ElevatedButton.icon(
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
                        style: GoogleFonts.inter(fontWeight: FontWeight.w600),
                      ),
                      style: ElevatedButton.styleFrom(
                        padding: const EdgeInsets.symmetric(vertical: 12),
                        backgroundColor: AppColors.primary.withValues(
                          alpha: 0.1,
                        ),
                        foregroundColor: AppColors.primary,
                        side: const BorderSide(color: AppColors.primary),
                      ),
                    ),
                  ),
                  if (_extractionError != null) ...[
                    const SizedBox(height: 12),
                    Container(
                      padding: const EdgeInsets.all(12),
                      decoration: BoxDecoration(
                        color: AppColors.error.withValues(alpha: 0.12),
                        borderRadius: BorderRadius.circular(8),
                        border: Border.all(
                          color: AppColors.error.withValues(alpha: 0.4),
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
                              style: GoogleFonts.inter(
                                fontSize: 12,
                                color: AppColors.error,
                              ),
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

            // ── Numeric Fields ──
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

            // ── Dropdown Fields ──
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

            // ── Submit Button ──
            _predictButton(),
            const SizedBox(height: 20),

            // ── Result / Error ──
            if (_result != null) _resultCard(),
            if (_error != null) _errorCard(),

            const SizedBox(height: 40),
          ],
        ),
      ),
    );
  }

  // ─── Helpers ───────────────────────────────────────────────────────

  Widget _sectionTitle(String title) {
    return Text(
      title,
      style: GoogleFonts.inter(
        fontSize: 15,
        fontWeight: FontWeight.w600,
        color: AppColors.accent,
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
    return Container(
      decoration: BoxDecoration(
        gradient: AppGradients.primary,
        borderRadius: BorderRadius.circular(14),
        boxShadow: [
          BoxShadow(
            color: AppColors.primary.withValues(alpha: 0.4),
            blurRadius: 16,
            offset: const Offset(0, 6),
          ),
        ],
      ),
      child: ElevatedButton(
        onPressed: _isLoading ? null : _predict,
        style: ElevatedButton.styleFrom(
          backgroundColor: Colors.transparent,
          shadowColor: Colors.transparent,
          padding: const EdgeInsets.symmetric(vertical: 18),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(14),
          ),
        ),
        child: _isLoading
            ? const SizedBox(
                height: 22,
                width: 22,
                child: CircularProgressIndicator(
                  color: Colors.white,
                  strokeWidth: 2.5,
                ),
              )
            : Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  const Icon(Icons.auto_awesome, size: 20),
                  const SizedBox(width: 10),
                  Text(
                    'Predict Car Price',
                    style: GoogleFonts.inter(
                      fontSize: 16,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ],
              ),
      ),
    );
  }

  Widget _resultCard() {
    return FadeTransition(
      opacity: _fadeAnim,
      child: Container(
        decoration: BoxDecoration(
          gradient: AppGradients.success,
          borderRadius: BorderRadius.circular(16),
          boxShadow: [
            BoxShadow(
              color: AppColors.success.withValues(alpha: 0.3),
              blurRadius: 20,
              offset: const Offset(0, 8),
            ),
          ],
        ),
        padding: const EdgeInsets.all(24),
        child: Column(
          children: [
            const Icon(
              Icons.check_circle_rounded,
              color: Colors.white,
              size: 48,
            ),
            const SizedBox(height: 12),
            Text(
              'Predicted Price',
              style: GoogleFonts.inter(
                fontSize: 14,
                fontWeight: FontWeight.w500,
                color: Colors.white70,
              ),
            ),
            const SizedBox(height: 6),
            Text(
              _result!.formattedPrice,
              style: GoogleFonts.inter(
                fontSize: 32,
                fontWeight: FontWeight.w800,
                color: Colors.white,
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
        color: AppColors.error.withValues(alpha: 0.12),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: AppColors.error.withValues(alpha: 0.4)),
      ),
      padding: const EdgeInsets.all(16),
      child: Row(
        children: [
          const Icon(Icons.error_outline, color: AppColors.error, size: 24),
          const SizedBox(width: 12),
          Expanded(
            child: Text(
              _error!,
              style: GoogleFonts.inter(fontSize: 13, color: AppColors.error),
            ),
          ),
        ],
      ),
    );
  }
}
