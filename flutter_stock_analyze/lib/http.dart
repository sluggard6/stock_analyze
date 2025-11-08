import 'package:flutter/foundation.dart' show kIsWeb;

import 'package:dio/dio.dart';

final dio = Dio();

// final String host = "http://localhost:5000";
final String host = "https://stock.myfile.live";

Future<String> fetchAnalysis(String stockCode) async {
  String url;
  if (kIsWeb) {
    url = '/analyze/$stockCode';
  } else {
    url = '$host/analyze/$stockCode';
  }
  final response = await dio.get(url);
  return response.data;
  // return getNumberLine(30);
}

String getNumberLine(int i) {
  String s = "";
  for (int j = 0; j < i; j++) {
    s += "$j\n";
  }
  return s;
}
