import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:path_provider/path_provider.dart';

import '../models/memory.dart';

/// Hatiralarin tek kaynagi.
///
/// Veritabani degil JSON: ses zaten diskte duran normal bir dosya, JSON
/// sadece dizin. Dizin bozulsa bile [_rebuildFromDisk] klasorleri tarayip
/// hatiralari geri getiriyor.
class MemoryStore extends ChangeNotifier {
  MemoryStore._();

  static final MemoryStore instance = MemoryStore._();

  /// `<app documents>` mutlak yolu. `late final` degil: dosya sistemine
  /// erisilemezse cokmek yerine bos kalip "dosya yok" davranisina dussun.
  String _rootPath = '';
  String get rootPath => _rootPath;

  /// `hatiralar` klasoru (gorece).
  static const String memoriesDirName = 'hatiralar';
  static const String indexFileName = 'hatiralar.json';

  List<Memory> _memories = <Memory>[];
  bool _loaded = false;

  /// En yeni hatira en ustte.
  List<Memory> get memories => List<Memory>.unmodifiable(_memories);
  bool get isLoaded => _loaded;
  bool get isEmpty => _loaded && _memories.isEmpty;

  File get _indexFile => File('$rootPath/$indexFileName');
  File get _backupFile => File('$rootPath/$indexFileName.yedek');
  Directory get memoriesDir => Directory('$rootPath/$memoriesDirName');

  /// Uygulama acilisinda bir kere cagrilir.
  Future<void> load() async {
    if (_loaded) return;
    final Directory docs = await getApplicationDocumentsDirectory();
    _rootPath = docs.path;
    if (!memoriesDir.existsSync()) {
      memoriesDir.createSync(recursive: true);
    }

    List<Memory>? parsed = await _readIndex(_indexFile);
    parsed ??= await _readIndex(_backupFile);

    if (parsed == null) {
      // Dizin okunamadi. Ses dosyalari yerinde olabilir; klasorlerden kurtar.
      parsed = await _rebuildFromDisk();
      if (parsed.isNotEmpty) {
        debugPrint('Hatira dizini bozuktu, ${parsed.length} hatira diskten kurtarildi.');
      }
    }

    _memories = parsed;
    _sort();
    _loaded = true;
    // Kurtarma yapildiysa saglam bir dizin yazalim.
    unawaited(_persist());
    notifyListeners();
  }

  Future<List<Memory>?> _readIndex(File file) async {
    try {
      if (!file.existsSync()) return null;
      final String raw = await file.readAsString();
      if (raw.trim().isEmpty) return null;
      final dynamic decoded = json.decode(raw);
      if (decoded is! List) return null;
      return decoded
          .whereType<Map<String, dynamic>>()
          .map(Memory.fromJson)
          .toList();
    } catch (e) {
      debugPrint('Hatira dizini okunamadi (${file.path}): $e');
      return null;
    }
  }

  /// JSON dizini kaybolduysa `hatiralar/<id>/` klasorlerinden yeniden kur.
  Future<List<Memory>> _rebuildFromDisk() async {
    final List<Memory> found = <Memory>[];
    try {
      if (!memoriesDir.existsSync()) return found;
      for (final FileSystemEntity entity in memoriesDir.listSync()) {
        if (entity is! Directory) continue;
        final String id = entity.uri.pathSegments
            .where((String s) => s.isNotEmpty)
            .last;
        final List<File> files = entity
            .listSync()
            .whereType<File>()
            .toList();
        final File? audio = files
            .where((File f) => f.path.endsWith('.m4a') || f.path.endsWith('.wav'))
            .firstOrNull;
        if (audio == null) continue;
        final DateTime when = audio.statSync().modified;
        found.add(
          Memory(
            id: id,
            title: 'Kurtarılan hatıra',
            createdAt: when,
            audioRelPath: '$memoriesDirName/$id/${_baseName(audio.path)}',
            durationMs: 0,
          ),
        );
      }
    } catch (e) {
      debugPrint('Diskten kurtarma basarisiz: $e');
    }
    return found;
  }

  static String _baseName(String path) => path.split(Platform.pathSeparator).last;

  void _sort() {
    _memories.sort((Memory a, Memory b) => b.createdAt.compareTo(a.createdAt));
  }

  Memory? byId(String id) {
    for (final Memory m in _memories) {
      if (m.id == id) return m;
    }
    return null;
  }

  /// Yeni hatira klasoru olusturur ve ses dosyasinin yazilacagi yolu doner.
  Future<({String id, String audioPath, String audioRelPath})> prepareNew(
    String id,
  ) async {
    final Directory dir = Directory('${memoriesDir.path}/$id');
    if (!dir.existsSync()) dir.createSync(recursive: true);
    const String fileName = 'ses.m4a';
    return (
      id: id,
      audioPath: '${dir.path}/$fileName',
      audioRelPath: '$memoriesDirName/$id/$fileName',
    );
  }

  Future<void> add(Memory memory) async {
    _memories.add(memory);
    _sort();
    notifyListeners();
    await _persist();
  }

  Future<void> update(Memory memory) async {
    final int i = _memories.indexWhere((Memory m) => m.id == memory.id);
    if (i < 0) return;
    _memories[i] = memory;
    notifyListeners();
    await _persist();
  }

  /// Hatirayi ve tum dosyalarini siler. Geri donusu yoktur.
  Future<void> delete(String id) async {
    _memories.removeWhere((Memory m) => m.id == id);
    notifyListeners();
    await _persist();
    try {
      final Directory dir = Directory('${memoriesDir.path}/$id');
      if (dir.existsSync()) await dir.delete(recursive: true);
    } catch (e) {
      debugPrint('Hatira klasoru silinemedi: $e');
    }
  }

  String absolute(String relPath) => '$rootPath/$relPath';

  /// Once gecici dosyaya yazip sonra tasiriz: yazma sirasinda pil biterse
  /// eski dizin bozulmamis kalir.
  Future<void> _persist() async {
    try {
      final String data = json.encode(
        _memories.map((Memory m) => m.toJson()).toList(),
      );
      final File tmp = File('${_indexFile.path}.gecici');
      await tmp.writeAsString(data, flush: true);
      if (_indexFile.existsSync()) {
        try {
          await _indexFile.copy(_backupFile.path);
        } catch (_) {
          // Yedek alinamazsa da asil yazma devam etsin.
        }
      }
      await tmp.rename(_indexFile.path);
    } catch (e) {
      debugPrint('Hatira dizini kaydedilemedi: $e');
    }
  }
}

extension _FirstOrNull<T> on Iterable<T> {
  T? get firstOrNull => isEmpty ? null : first;
}
