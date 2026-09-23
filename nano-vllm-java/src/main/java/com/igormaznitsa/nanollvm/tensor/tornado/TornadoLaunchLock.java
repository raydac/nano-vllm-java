package com.igormaznitsa.nanollvm.tensor.tornado;

import java.util.concurrent.locks.ReentrantLock;

/**
 * Serializes TornadoVM launches. The device runtime is not safe for overlapping executes.
 */
final class TornadoLaunchLock {

  private static final ReentrantLock LOCK = new ReentrantLock();

  private TornadoLaunchLock() {
  }

  static void lock() {
    LOCK.lock();
  }

  static void unlock() {
    LOCK.unlock();
  }
}
