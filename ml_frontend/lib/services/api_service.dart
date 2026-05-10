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
  // Production URL for Render backend
  static const String _productionUrl =
      'https://asset-price-predictor-api.onrender.com';

  static String get baseUrl {
    if (kReleaseMode) {
      return _productionUrl;
    }

    if (kIsWeb) {
      return 'http://127.0.0.1:8000';
    }
    // Android emulator → http://10.0.2.2:8000
    // iOS simulator    → http://127.0.0.1:8000
    return 'http://127.0.0.1:8000';
  }

  static String userMessageFromError(Object error, {required String fallback}) {
    final raw = error.toString().replaceFirst('Exception: ', '').trim();
    if (raw.isEmpty) return fallback;

    final lower = raw.toLowerCase();
    if (lower.contains('socketexception') ||
        lower.contains('failed host lookup') ||
        lower.contains('connection refused') ||
        lower.contains('cannot connect') ||
        lower.contains('network is unreachable')) {
      return 'We could not reach the server. Please make sure the backend is running and try again.';
    }

    if (lower.contains('timed out')) {
      return 'The request took too long. Please try again.';
    }

    if (lower.contains('404') ||
        lower.contains('500') ||
        lower.contains('502') ||
        lower.contains('503')) {
      return 'The service is temporarily unavailable. Please try again in a moment.';
    }

    if (lower.contains('validation') || lower.contains('invalid')) {
      return 'Some details need to be corrected before continuing.';
    }

    return raw;
  }

  // ─── Dropdown Options ──────────────────────────────────────────────

  Future<DropdownOptions> getDropdownOptions() async {
    final url = Uri.parse('$baseUrl/options');
    try {
      final response = await http.get(url).timeout(const Duration(seconds: 10));
      if (response.statusCode == 200) {
        return DropdownOptions.fromJson(jsonDecode(response.body));
      } else {
        throw Exception(
          'We could not load the app data. Please try again in a moment.',
        );
      }
    } catch (e) {
      throw Exception(
        userMessageFromError(
          e,
          fallback:
              'We could not load the app data. Please check your connection and try again.',
        ),
      );
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
        throw Exception(
          'We could not generate a car price right now. Please review the details and try again.',
        );
      }
    } catch (e) {
      throw Exception(
        userMessageFromError(
          e,
          fallback:
              'We could not generate a car price right now. Please try again.',
        ),
      );
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
        throw Exception(
          'We could not generate a house price right now. Please review the details and try again.',
        );
      }
    } catch (e) {
      throw Exception(
        userMessageFromError(
          e,
          fallback:
              'We could not generate a house price right now. Please try again.',
        ),
      );
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
        throw Exception(
          'We could not read details from that listing. You can fill the form manually.',
        );
      }
    } catch (e) {
      throw Exception(
        userMessageFromError(
          e,
          fallback:
              'We could not read details from that listing. You can fill the form manually.',
        ),
      );
    }
  }

  // ─── House Field Extraction (NER from URL) ───────────────────────

  Future<HouseFieldsOutput> extractHouseFields(String url) async {
    final apiUrl = Uri.parse('$baseUrl/extract/house-fields');
    try {
      final response = await http
          .post(
            apiUrl,
            headers: {'Content-Type': 'application/json'},
            body: jsonEncode({'url': url}),
          )
          .timeout(const Duration(seconds: 30));

      if (response.statusCode == 200) {
        return HouseFieldsOutput.fromJson(jsonDecode(response.body));
      } else {
        throw Exception(
          'We could not read details from that listing. You can fill the form manually.',
        );
      }
    } catch (e) {
      throw Exception(
        userMessageFromError(
          e,
          fallback:
              'We could not read details from that listing. You can fill the form manually.',
        ),
      );
    }
  }
}
