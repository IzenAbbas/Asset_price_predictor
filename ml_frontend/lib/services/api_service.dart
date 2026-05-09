/// HTTP service layer for communicating with the FastAPI backend.
///
/// All network calls are centralised here so the UI code stays clean.
library;

import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import '../models/prediction_model.dart';

class ApiService {
  // ─── Base URL Configuration ────────────────────────────────────────
  // To deploy: Replace with your Render URL (e.g., https://your-app.onrender.com)
  static const String _productionUrl = 'YOUR_RENDER_URL';

  static String get baseUrl {
    if (kReleaseMode && _productionUrl != 'YOUR_RENDER_URL') {
      return _productionUrl;
    }

    if (kIsWeb) {
      return 'http://127.0.0.1:8000';
    }
    // Android emulator → http://10.0.2.2:8000
    // iOS simulator    → http://127.0.0.1:8000
    return 'http://127.0.0.1:8000';
  }

  // ─── Dropdown Options ──────────────────────────────────────────────

  Future<DropdownOptions> getDropdownOptions() async {
    final url = Uri.parse('$baseUrl/options');
    try {
      final response = await http.get(url).timeout(const Duration(seconds: 10));
      if (response.statusCode == 200) {
        return DropdownOptions.fromJson(jsonDecode(response.body));
      } else {
        throw Exception('Failed to load options: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('Cannot connect to API: $e');
    }
  }

  // ─── Car Price Prediction ──────────────────────────────────────────

  Future<CarPredictionOutput> predictCarPrice(CarPredictionInput input) async {
    final url = Uri.parse('$baseUrl/predict/car');
    try {
      final response = await http
          .post(
            url,
            headers: {'Content-Type': 'application/json'},
            body: jsonEncode(input.toJson()),
          )
          .timeout(const Duration(seconds: 15));

      if (response.statusCode == 200) {
        return CarPredictionOutput.fromJson(jsonDecode(response.body));
      } else {
        final detail = jsonDecode(response.body)['detail'] ?? response.body;
        throw Exception('Prediction failed: $detail');
      }
    } catch (e) {
      throw Exception('Car prediction error: $e');
    }
  }

  // ─── House Price Prediction ────────────────────────────────────────

  Future<HousePredictionOutput> predictHousePrice(
    HousePredictionInput input,
  ) async {
    final url = Uri.parse('$baseUrl/predict/house');
    try {
      final response = await http
          .post(
            url,
            headers: {'Content-Type': 'application/json'},
            body: jsonEncode(input.toJson()),
          )
          .timeout(const Duration(seconds: 15));

      if (response.statusCode == 200) {
        return HousePredictionOutput.fromJson(jsonDecode(response.body));
      } else {
        final detail = jsonDecode(response.body)['detail'] ?? response.body;
        throw Exception('Prediction failed: $detail');
      }
    } catch (e) {
      throw Exception('House prediction error: $e');
    }
  }

  // ─── Vehicle Field Extraction (NER from URL) ───────────────────────

  Future<VehicleFieldsOutput> extractVehicleFields(String url) async {
    final apiUrl = Uri.parse('$baseUrl/extract/vehicle-fields');
    try {
      final response = await http
          .post(
            apiUrl,
            headers: {'Content-Type': 'application/json'},
            body: jsonEncode({'url': url}),
          )
          .timeout(const Duration(seconds: 30));

      if (response.statusCode == 200) {
        return VehicleFieldsOutput.fromJson(jsonDecode(response.body));
      } else {
        final body = jsonDecode(response.body);
        final detail = body['detail'] ?? response.body;
        throw Exception(detail);
      }
    } catch (e) {
      throw Exception('$e');
    }
  }
}
