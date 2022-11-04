/*
 * Copyright (c) CISIAD, UNED, Spain,  2019. Licensed under the GPLv3 licence
 * Unless required by applicable law or agreed to in writing,
 * this code is distributed on an "AS IS" basis,
 * WITHOUT WARRANTIES OF ANY KIND.
 */
package org.openmarkov.gui.dialog.io;

import java.util.HashMap;

@SuppressWarnings("serial") public class DBReaderFileChooser extends DBFileChooser {
	public DBReaderFileChooser(boolean acceptAllFiles) {
		super(acceptAllFiles);
		HashMap<String, String> writersInfo = caseDbManager.getAllReaders();
		for (String extension : writersInfo.keySet()) {
			addChoosableFileFilter(new FileFilterAll(extension, writersInfo.get(extension)));
		}
	}

	public DBReaderFileChooser() {
		this(false);
	}
}
