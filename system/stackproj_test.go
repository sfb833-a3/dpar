// Copyright 2015 The dpar Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package system

import (
	"testing"
)

func TestStackProjectiveSystem(t *testing.T) {
	testSystem(t, NewStackProjective(), NewStackProjectiveOracle, ProjectiveData)
}
